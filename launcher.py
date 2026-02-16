import threading
import queue
import sys
import time

from predict import run_detection
from prediction_agent import run_agent


def main():
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("="*70, flush=True)
    print("AI DETECTION SYSTEM - LAUNCHER", flush=True)
    print("="*70, flush=True)
    print("Starting Detection and Alert System...", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Step 1: Create shared queue
    print("[LAUNCHER] Step 1/4: Creating event queue...", flush=True)
    event_queue = queue.Queue()
    print("[LAUNCHER] ✓ Queue created\n", flush=True)
    
    # Step 2: Create synchronization event
    print("[LAUNCHER] Step 2/4: Creating synchronization event...", flush=True)
    agent_ready = threading.Event()
    print("[LAUNCHER] ✓ Sync event created\n", flush=True)
    
    # Step 3: Start and initialize agent thread
    print("[LAUNCHER] Step 3/4: Starting agent thread...", flush=True)
    agent_thread = threading.Thread(
        target=run_agent, 
        args=(event_queue, agent_ready), 
        name="AlertAgent", 
        daemon=True
    )
    agent_thread.start()
    print("[LAUNCHER] ✓ Thread started, waiting for agent to initialize...\n", flush=True)
    
    # BLOCK HERE until agent is fully ready
    print("[LAUNCHER] ⏳ Blocking until agent signals ready...", flush=True)
    agent_ready.wait()  # THIS BLOCKS UNTIL AGENT CALLS agent_ready.set()
    print("[LAUNCHER] ✓ Agent confirmed ready!\n", flush=True)
    
    # Step 4: NOW start detection (agent is guaranteed to be listening)
    print("[LAUNCHER] Step 4/4: Starting detection process...", flush=True)
    print("[LAUNCHER] ✓ Agent is listening, starting video processing NOW\n", flush=True)
    print("="*70 + "\n", flush=True)
    
    try:
        run_detection(event_queue)
    
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Interrupted by user")
    
    finally:
        # Cleanup
        print("\n[LAUNCHER] Shutting down...")
        
        # Wait for agent to finish processing remaining events
        print("[LAUNCHER] Waiting for agent to finish...")
        event_queue.join()
        
        print("[LAUNCHER] System stopped")
        print("="*70)


if __name__ == "__main__":
    main()