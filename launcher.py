import threading
import queue
import sys
import time

from video_classifier import classify_video
from predict import run_detection
from prediction_agent import AlertAgent


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 70, flush=True)
    print("AI DETECTION SYSTEM - LAUNCHER", flush=True)
    print("=" * 70, flush=True)
    print("Starting Detection and Alert System...", flush=True)
    print("=" * 70 + "\n", flush=True)

    # ── Step 1: Queue and sync event ────────────────────────────
    print("[LAUNCHER] Step 1/5: Creating event queue and sync event...", flush=True)
    event_queue = queue.Queue()
    agent_ready = threading.Event()
    print("[LAUNCHER] ✓ Done\n", flush=True)

    # ── Step 2: Initialize agent (the brain) ────────────────────
    print("[LAUNCHER] Step 2/5: Initializing agent (system brain)...", flush=True)
    agent = AlertAgent(ready_event=agent_ready)
    print("[LAUNCHER] ✓ Agent initialized\n", flush=True)

    # ── Step 3: Run video classifier ────────────────────────────
    print("[LAUNCHER] Step 3/5: Running video pre-classifier...", flush=True)
    classifier_output = classify_video()
    print(f"[LAUNCHER] ✓ Classifier done → detected: {classifier_output['detected']}\n", flush=True)

    # ── Step 4: Agent decides which models to run ────────────────
    print("[LAUNCHER] Step 4/5: Agent making model routing decision...", flush=True)
    selected_models = agent.decide_models(classifier_output)
    model_names = [key for key, _ in selected_models]
    print(f"[LAUNCHER] ✓ Agent selected models: {model_names}\n", flush=True)

    # ── Step 5: Start agent event loop in background ────────────
    print("[LAUNCHER] Step 5/5: Starting agent event loop thread...", flush=True)
    agent_thread = threading.Thread(
        target=agent.run,
        args=(event_queue,),
        name="AlertAgent",
        daemon=True
    )
    agent_thread.start()

    print("[LAUNCHER] ⏳ Waiting for agent event loop to be ready...", flush=True)
    agent_ready.wait()
    print("[LAUNCHER] ✓ Agent is listening\n", flush=True)

    # ── Start detection with agent-chosen models ─────────────────
    print(f"[LAUNCHER] Starting detection → models: {model_names}", flush=True)
    print("=" * 70 + "\n", flush=True)

    try:
        run_detection(event_queue, selected_models)

    except KeyboardInterrupt:
        print("\n[LAUNCHER] Interrupted by user")

    finally:
        print("\n[LAUNCHER] Shutting down...")
        print("[LAUNCHER] Waiting for agent to finish processing remaining events...")
        event_queue.join()
        print("[LAUNCHER] System stopped")
        print("=" * 70)


if __name__ == "__main__":
    main()