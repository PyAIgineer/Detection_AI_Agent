# ==================== PREDICTION_AGENT.PY ====================
# Agent is the BRAIN of the entire system.
# Responsibility 1: Receive classifier JSON → decide which YOLO model to run
# Responsibility 2: Receive detection events → decide severity → send alerts

import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from openai import OpenAI
import queue

from config import MODEL_CONFIGS

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ==================== ROUTING SYSTEM PROMPT ====================
ROUTING_SYSTEM_PROMPT = """You are an AI model router for a surveillance detection system.

You receive a JSON output from a video classifier that lists what was detected in the video.

Available individual models (each detects ONE category):
- "deer"       : detects deer
- "elephant"   : detects elephant
- "fire_smoke" : detects fire and smoke
- "leopard"    : detects leopard
- "tiger"      : detects tiger

Rules:
- Select ALL models that match what was detected. You can select multiple.
- If "fire" or "smoke" detected → include "fire_smoke"
- If "deer" detected → include "deer"
- If "elephant" detected → include "elephant"
- If "leopard" detected → include "leopard"
- If "tiger" detected → include "tiger"
- If "person" detected but no matching animal model → still select the closest animal model (e.g. if deer + person, select "deer")
- If nothing clear detected → select ["elephant"] as safe default

Respond ONLY in this JSON format:
{
    "model_keys": ["key1", "key2"],
    "reason": "short explanation"
}"""


# ==================== ALERT SYSTEM PROMPT ====================
ALERT_SYSTEM_PROMPT = """You are an intelligent security monitoring agent.

Your job is to analyze raw detection events from a camera feed and decide:
1. How severe the situation is
2. What action to take
3. Write an appropriate alert message

You will receive raw detection data and must respond ONLY in this JSON format:
{
    "severity": "low | medium | high | critical",
    "action": "log_only | email | escalate",
    "subject": "email subject line",
    "message": "2-3 sentence professional alert message",
    "reason": "why you made this decision"
}

Severity and action rules:
- fire, smoke detected → critical → escalate
- dangerous/wild animal (elephant, leopard, tiger, lion, bear) entering boundary → critical → escalate
- deer entering boundary → high → email
- person entering restricted boundary → high → email
- person detected in frame (no boundary) → medium → email
- animal detected in frame (no boundary) → medium → email
- any detection with confidence < 0.5 → low → log_only
- unknown or unclear object → low → log_only

Action definitions:
- log_only: not urgent, just record it
- email: send email alert to security team
- escalate: send URGENT email, needs immediate attention

Always make a decision. Never ask for clarification. Only return the JSON."""


class AlertAgent:
    """
    The brain of the detection system.
    - Decides which YOLO model to run based on classifier output.
    - Processes detection events and sends appropriate alerts.
    """

    def __init__(self, ready_event=None):
        # Email config
        self.email_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
        self.email_port = os.getenv('EMAIL_PORT', '587')
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', 'sameermankar1234@gmail.com')

        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

        self.email_enabled = bool(self.email_user and self.email_password)
        self.ready_event = ready_event

        print("[AGENT] Initializing Alert Agent (System Brain)...", flush=True)
        print(f"[AGENT] Email : {'Enabled' if self.email_enabled else 'Disabled'}", flush=True)
        print(f"[AGENT] GPT   : {'Enabled' if self.openai_client else 'Disabled'}", flush=True)
        print(f"[AGENT] Recipient: {self.recipient_email}", flush=True)
        print(f"[AGENT] Initialization complete\n", flush=True)


    # ==================== BRAIN: MODEL ROUTING ====================
    def decide_models(self, classifier_output):
        """
        Called by launcher BEFORE detection starts.
        Returns a LIST of (model_key, model_config) tuples — one per detected category.
        All selected models will run together on every frame.
        """
        print(f"\n{'='*60}", flush=True)
        print("[AGENT-BRAIN] Classifier output received:", flush=True)
        print(f"[AGENT-BRAIN] {json.dumps(classifier_output)}", flush=True)
        print("[AGENT-BRAIN] Making model routing decision...", flush=True)

        if self.openai_client:
            model_keys = self._llm_routing_decision(classifier_output)
        else:
            model_keys = self._fallback_routing_decision(classifier_output)

        # Validate and resolve all keys
        valid_keys = []
        for key in model_keys:
            if key in MODEL_CONFIGS:
                valid_keys.append(key)
            else:
                print(f"[AGENT-BRAIN] WARNING: Unknown model key '{key}', skipping", flush=True)

        if not valid_keys:
            print("[AGENT-BRAIN] WARNING: No valid models selected, defaulting to 'elephant'", flush=True)
            valid_keys = ["elephant"]

        selected = [(key, MODEL_CONFIGS[key]) for key in valid_keys]

        print(f"[AGENT-BRAIN] ✓ Models selected ({len(selected)}):", flush=True)
        for key, cfg in selected:
            print(f"[AGENT-BRAIN]   - {key} → {cfg['model_path']} | classes: {cfg['target_classes']}", flush=True)
        print(f"{'='*60}\n", flush=True)

        return selected

    def _llm_routing_decision(self, classifier_output):
        """Use GPT to decide which models to run — returns list of model keys"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classifier output:\n{json.dumps(classifier_output, indent=2)}"}
                ],
                max_tokens=150,
                temperature=0.1
            )

            raw = response.choices[0].message.content.strip()
            decision = json.loads(raw)

            keys = decision.get("model_keys", [])
            print(f"[AGENT-BRAIN] LLM routing → {keys} | Reason: {decision.get('reason', '')}", flush=True)
            return keys

        except Exception as e:
            print(f"[AGENT-BRAIN] GPT routing error: {e} - using fallback", flush=True)
            return self._fallback_routing_decision(classifier_output)

    def _fallback_routing_decision(self, classifier_output):
        """Rule-based routing — returns list of model keys"""
        detected = [d.lower() for d in classifier_output.get("detected", [])]

        animal_map = {
            "deer":     "deer",
            "elephant": "elephant",
            "leopard":  "leopard",
            "tiger":    "tiger",
        }

        keys = []

        if "fire" in detected or "smoke" in detected:
            keys.append("fire_smoke")

        for animal, key in animal_map.items():
            if animal in detected:
                keys.append(key)

        if not keys:
            keys = ["elephant"]  # safe default
            print("[AGENT-BRAIN] Fallback: nothing matched, defaulting to 'elephant'", flush=True)
        else:
            print(f"[AGENT-BRAIN] Fallback routing → {keys}", flush=True)

        return keys


    # ==================== BRAIN: ALERT DECISION ====================
    def analyze_and_decide(self, event):
        """LLM receives raw detection event and decides severity, action, message"""
        if not self.openai_client:
            return self._fallback_alert_decision(event)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": ALERT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Raw detection event:\n{json.dumps(event, indent=2)}"}
                ],
                max_tokens=300,
                temperature=0.2
            )

            raw = response.choices[0].message.content.strip()
            decision = json.loads(raw)

            print(f"[AGENT] LLM Alert Decision:", flush=True)
            print(f"         Severity : {decision['severity'].upper()}", flush=True)
            print(f"         Action   : {decision['action']}", flush=True)
            print(f"         Reason   : {decision['reason']}", flush=True)

            return decision

        except json.JSONDecodeError as e:
            print(f"[AGENT] JSON parse error: {e} - using fallback", flush=True)
            return self._fallback_alert_decision(event)

        except Exception as e:
            print(f"[AGENT] GPT error: {e} - using fallback", flush=True)
            return self._fallback_alert_decision(event)

    def _fallback_alert_decision(self, event):
        """Rule-based fallback when GPT unavailable"""
        class_name = event.get('class_name', '').lower()
        confidence = event.get('confidence', 1.0)
        has_boundary = 'direction' in event

        dangerous_animals = ['elephant', 'leopard', 'tiger', 'lion', 'bear']

        if confidence < 0.5:
            severity, action = 'low', 'log_only'
        elif class_name in ['fire', 'smoke']:
            severity, action = 'critical', 'escalate'
        elif class_name in dangerous_animals and has_boundary:
            severity, action = 'critical', 'escalate'
        elif has_boundary:
            severity, action = 'high', 'email'
        else:
            severity, action = 'medium', 'email'

        message = f"{class_name.upper()} detected"
        if has_boundary:
            message += f" {event['direction']} the boundary"
        message += f" at {event['timestamp']}."

        return {
            "severity": severity,
            "action": action,
            "subject": f"{'URGENT' if action == 'escalate' else 'Alert'}: {class_name.upper()} Detected",
            "message": message,
            "reason": "Fallback rule-based decision (GPT unavailable)"
        }


    # ==================== ACTIONS ====================
    def send_email(self, decision, event):
        """Send email alert"""
        if not self.email_enabled:
            print(f"[AGENT] Email disabled - logged: {decision['subject']}", flush=True)
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.recipient_email
            msg['Subject'] = decision['subject']

            summary = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in event.items()])
            urgency = "URGENT - IMMEDIATE ACTION REQUIRED\n\n" if decision['action'] == 'escalate' else ""

            body = f"""
AI Security Monitoring Agent Alert
{'='*60}

{urgency}{decision['message']}

{'='*60}
Severity     : {decision['severity'].upper()}
Action Taken : {decision['action'].upper()}
Agent Reason : {decision['reason']}
{'='*60}
Raw Detection Data:
{summary}
{'='*60}

This is an automated alert from the AI Detection Agent.
"""
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_host, int(self.email_port))
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()

            print(f"[AGENT] ✓ Email sent: {decision['subject']}", flush=True)
            return True

        except Exception as e:
            print(f"[AGENT] Email error: {e}", flush=True)
            return False

    def log_only(self, decision, event):
        """Log event without sending alert"""
        print(f"[AGENT] Logged (no alert): {event.get('class_name')} | Reason: {decision['reason']}", flush=True)


    # ==================== PROCESS DETECTION EVENT ====================
    def process_event(self, event):
        """
        Responsibility 2: receive raw detection event, decide severity, execute action.
        """
        print(f"\n{'='*60}", flush=True)
        print(f"[AGENT] Detection event received: {json.dumps(event)}", flush=True)
        print(f"{'='*60}", flush=True)

        decision = self.analyze_and_decide(event)
        action = decision.get('action', 'log_only')

        if action == 'log_only':
            self.log_only(decision, event)
        elif action == 'email':
            print(f"[AGENT] Sending email alert...", flush=True)
            self.send_email(decision, event)
        elif action == 'escalate':
            print(f"[AGENT] ⚠ ESCALATING - Critical situation!", flush=True)
            self.send_email(decision, event)
        else:
            print(f"[AGENT] Unknown action '{action}' - defaulting to log", flush=True)
            self.log_only(decision, event)

        print(f"{'='*60}\n", flush=True)


    # ==================== EVENT LOOP ====================
    def run(self, event_queue):
        """Main agent event loop - listens for detection events from predict.py"""
        print("[AGENT] Entering detection event monitoring loop...", flush=True)

        try:
            if self.ready_event:
                print("[AGENT] Signaling ready to launcher...", flush=True)
                self.ready_event.set()
                print("[AGENT] READY - Listening for detection events...\n", flush=True)

            while True:
                event = event_queue.get()

                if event is None:
                    event_queue.task_done()
                    print("[AGENT] Received shutdown signal", flush=True)
                    break

                self.process_event(event)
                event_queue.task_done()

        except KeyboardInterrupt:
            print("\n[AGENT] Interrupted by user", flush=True)

        finally:
            print("[AGENT] Agent stopped", flush=True)


# ==================== ENTRY POINT FOR THREAD ====================
def run_agent(event_queue, ready_event=None):
    agent = AlertAgent(ready_event=ready_event)
    agent.run(event_queue)


if __name__ == "__main__":
    print("[INFO] Agent cannot run standalone")
    print("[INFO] Use launcher.py to start the complete system")