import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from openai import OpenAI
import queue

# Load environment variables
load_dotenv()

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ==================== AGENT SYSTEM PROMPT ====================
AGENT_SYSTEM_PROMPT = """You are an intelligent security monitoring agent.

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
- dangerous/wild animal (elephant, leopard, lion, bear) entering boundary → critical → escalate
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
    """Agent that uses LLM to decide severity and action for each detection event"""

    def __init__(self, ready_event=None):
        # Email Configuration
        self.email_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
        self.email_port = os.getenv('EMAIL_PORT', '587')
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', 'sameermankar1234@gmail.com')

        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Validation
        self.email_enabled = bool(self.email_user and self.email_password)
        self.ready_event = ready_event

        print("[AGENT] Initializing Alert Agent...", flush=True)
        print(f"[AGENT] Email: {'Enabled' if self.email_enabled else 'Disabled'}", flush=True)
        print(f"[AGENT] GPT: {'Enabled' if self.openai_client else 'Disabled'}", flush=True)
        print(f"[AGENT] Recipient: {self.recipient_email}", flush=True)
        print(f"[AGENT] Initialization complete\n", flush=True)


    # ==================== AGENT DECISION ====================
    def analyze_and_decide(self, event):
        """LLM receives raw event and decides severity, action, message"""
        if not self.openai_client:
            print("[AGENT] GPT unavailable - using fallback decision", flush=True)
            return self._fallback_decision(event)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Raw detection event:\n{json.dumps(event, indent=2)}"}
                ],
                max_tokens=300,
                temperature=0.2
            )

            raw = response.choices[0].message.content.strip()
            decision = json.loads(raw)

            print(f"[AGENT] LLM Decision:", flush=True)
            print(f"         Severity : {decision['severity'].upper()}", flush=True)
            print(f"         Action   : {decision['action']}", flush=True)
            print(f"         Reason   : {decision['reason']}", flush=True)

            return decision

        except json.JSONDecodeError as e:
            print(f"[AGENT] JSON parse error: {e} - using fallback", flush=True)
            return self._fallback_decision(event)

        except Exception as e:
            print(f"[AGENT] GPT error: {e} - using fallback", flush=True)
            return self._fallback_decision(event)


    # ==================== FALLBACK ====================
    def _fallback_decision(self, event):
        """Rule-based fallback when GPT is unavailable"""
        class_name = event.get('class_name', '').lower()
        confidence = event.get('confidence', 1.0)
        has_boundary = 'direction' in event

        if confidence < 0.5:
            severity, action = 'low', 'log_only'
        elif class_name in ['fire', 'smoke']:
            severity, action = 'critical', 'escalate'
        elif class_name in ['elephant', 'leopard', 'lion', 'bear', 'tiger'] and has_boundary:
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

            print(f"[AGENT] Email sent: {decision['subject']}", flush=True)
            return True

        except Exception as e:
            print(f"[AGENT] Email error: {e}", flush=True)
            return False

    def log_only(self, decision, event):
        """Log event without sending alert"""
        print(f"[AGENT] Logged (no alert): {event.get('class_name')} | Reason: {decision['reason']}", flush=True)


    # ==================== PROCESS EVENT ====================
    def process_event(self, event):
        """
        Agent receives raw detection facts.
        LLM decides severity and action.
        Agent executes the decision.
        """
        print(f"\n{'='*60}", flush=True)
        print(f"[AGENT] Raw event received: {json.dumps(event)}", flush=True)
        print(f"{'='*60}", flush=True)

        decision = self.analyze_and_decide(event)
        action = decision.get('action', 'log_only')

        if action == 'log_only':
            self.log_only(decision, event)

        elif action == 'email':
            print(f"[AGENT] Sending email alert...", flush=True)
            self.send_email(decision, event)

        elif action == 'escalate':
            print(f"[AGENT] ESCALATING - Critical situation!", flush=True)
            self.send_email(decision, event)
            # Future: SMS, webhook, etc.

        else:
            print(f"[AGENT] Unknown action '{action}' - defaulting to log", flush=True)
            self.log_only(decision, event)

        print(f"{'='*60}\n", flush=True)


    # ==================== MAIN LOOP ====================
    def run(self, event_queue):
        """Main agent loop"""
        print("[AGENT] Entering monitoring loop...", flush=True)

        try:
            if self.ready_event:
                print("[AGENT] Signaling ready to launcher...", flush=True)
                self.ready_event.set()
                print("[AGENT] READY - Listening for events...\n", flush=True)

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


# ==================== STANDALONE MODE ====================
def run_agent(event_queue, ready_event=None):
    agent = AlertAgent(ready_event=ready_event)
    agent.run(event_queue)


if __name__ == "__main__":
    print("[INFO] Agent cannot run standalone")
    print("[INFO] Use launcher.py to start the complete system\n")