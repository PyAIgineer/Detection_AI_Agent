import os
import sys
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


class AlertAgent:
    """Independent agent for processing detection events and sending alerts"""
    
    def __init__(self, ready_event=None):
        # Email Configuration
        self.email_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
        self.email_port = os.getenv('EMAIL_PORT', '587')
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', 'sameermankar1234@gmail.com')
        
        # OpenAI Configuration
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Validation
        self.email_enabled = bool(self.email_user and self.email_password)
        
        # Store ready event
        self.ready_event = ready_event
        
        print("[AGENT] ⚙️  Initializing Alert Agent...", flush=True)
        print(f"[AGENT] ✓ Email: {'Enabled' if self.email_enabled else 'Disabled (credentials not found)'}", flush=True)
        print(f"[AGENT] ✓ GPT: {'Enabled' if self.openai_client else 'Disabled (API key not found)'}", flush=True)
        print(f"[AGENT] ✓ Recipient: {self.recipient_email}", flush=True)
        print(f"[AGENT] ✓ Initialization complete\n", flush=True)
    
    def generate_alert_message(self, event):
        """Generate alert message using GPT-3.5-turbo"""
        if not self.openai_client:
            return self._fallback_message(event)
        
        try:
            # Build prompt based on event type
            if event['type'] == 'detection':
                prompt = f"""Generate a concise alert message for the following detection:

Detection Type: {event['class_name']}
Number of Objects: {event['count']}
Timestamp: {event['timestamp']}
Frame: {event['frame']}

Create a brief, professional alert message (2-3 sentences) about this detection event."""
            
            else:  # crossing
                prompt = f"""Generate a concise alert message for the following boundary crossing:

Object Type: {event['class_name']}
Direction: {event['direction']}
Timestamp: {event['timestamp']}
Frame: {event['frame']}

Create a brief, professional alert message (2-3 sentences) about this boundary crossing event."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a security alert system. Generate concise, professional alert messages."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"[AGENT] GPT generation failed: {e}")
            return self._fallback_message(event)
    
    def _fallback_message(self, event):
        """Generate fallback message when GPT is unavailable"""
        if event['type'] == 'detection':
            return f"{event['class_name'].upper()} Detection Alert: {event['count']} object(s) detected at {event['timestamp']}."
        else:
            return f"{event['class_name'].upper()} Boundary Crossing Alert: Object {event['direction']} the boundary at {event['timestamp']}."
    
    def send_email(self, subject, message, event):
        """Send email alert"""
        if not self.email_enabled:
            print(f"[AGENT] Email disabled - Alert logged: {subject}")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            # Build email body
            if event['type'] == 'detection':
                summary = f"""Detection Summary:
- Object Type: {event['class_name'].upper()}
- Count: {event['count']}
- Timestamp: {event['timestamp']}
- Frame: {event['frame']}"""
            else:
                summary = f"""Crossing Summary:
- Object Type: {event['class_name'].upper()}
- Direction: {event['direction'].capitalize()}
- Timestamp: {event['timestamp']}
- Frame: {event['frame']}"""
            
            body = f"""
Detection Alert
{'='*60}

{message}

{'='*60}
{summary}
{'='*60}

This is an automated alert from the AI Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_host, int(self.email_port))
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()
            
            print(f"[AGENT] ✅ Email sent: {subject}", flush=True)
            return True
            
        except Exception as e:
            print(f"[AGENT] ❌ Email error: {e}", flush=True)
            return False
    
    def process_event(self, event):
        """Process a single detection event"""
        print(f"\n{'='*60}", flush=True)
        print(f"[AGENT] 🔔 Processing: {event['type'].upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Generate alert message
        alert_message = self.generate_alert_message(event)
        
        # Prepare subject
        if event['type'] == 'detection':
            subject = f"🚨 {event['class_name'].upper()} Detection Alert"
        else:
            subject = f"⚠️ {event['class_name'].upper()} Boundary Crossing"
        
        # Send email
        self.send_email(subject, alert_message, event)
        
        print(f"{'='*60}\n")
    
    def run(self, event_queue):
        """Main agent loop - monitors queue for events"""
        print("[AGENT] 🎯 Entering monitoring loop...", flush=True)
        
        try:
            # Signal ready RIGHT BEFORE entering blocking state
            if self.ready_event:
                print("[AGENT] 📡 About to signal ready to launcher...", flush=True)
                self.ready_event.set()
                print("[AGENT] ✅ READY SIGNAL SENT - Agent is now listening for events", flush=True)
                print("[AGENT] 👂 Blocking on queue.get() - waiting for events...\n", flush=True)
            else:
                print("[AGENT] ⚠️  No ready_event provided, starting monitoring anyway\n", flush=True)
            
            while True:
                # Block until event is available
                event = event_queue.get()
                
                # None signals shutdown
                if event is None:
                    event_queue.task_done()
                    print("[AGENT] Received shutdown signal")
                    break
                
                # Process the event
                self.process_event(event)
                
                # Mark task as done
                event_queue.task_done()
        
        except KeyboardInterrupt:
            print("\n[AGENT] Interrupted by user")
        
        finally:
            print("[AGENT] Agent stopped")


# ==================== STANDALONE MODE ====================
def run_agent(event_queue, ready_event=None):
    """Run agent as a separate thread/process"""
    agent = AlertAgent(ready_event=ready_event)
    agent.run(event_queue)


if __name__ == "__main__":
    print("[INFO] Agent cannot run standalone")
    print("[INFO] Use launcher.py to start the complete system\n")