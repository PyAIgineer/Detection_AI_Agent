# ==================== VIDEO_CLASSIFIER.PY ====================
# Extracts frames 1, 15, 30 from input video and sends them to
# Groq vision LLM to identify what's present before running YOLO.

import sys
import cv2
import json
import base64
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

from config import VIDEO_PATH, CLASSIFIER_FRAMES, GROQ_MODEL

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ==================== PROMPT ====================
CLASSIFIER_PROMPT = """You are a video scene classifier. You will receive 3 frames from a surveillance video.

Your job is to identify ONLY what is CLEARLY VISIBLE in these frames.

Detectable categories:
- fire
- smoke
- leopard
- tiger
- deer
- elephant
- person

Rules:
- Only include classes you can clearly see. Do NOT guess.
- If there are animals AND people, include both.
- If there is a boundary/fence AND animals/people near it, add "boundary_crossing" to detected list.
- "boundary_crossing" means a physical fence/wall/boundary line is visible AND objects are near/crossing it.

Respond ONLY in this JSON format with no extra text:
{
    "detected": ["class1", "class2"],
    "scenario": "brief one-line description of what's happening in the video"
}"""


# ==================== FRAME EXTRACTION ====================
def extract_frames(video_path, frame_numbers):
    """Extract specific frames from video and return as base64 encoded JPEGs"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[CLASSIFIER] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[CLASSIFIER] Video opened: {video_path} | Total frames: {total_frames}", flush=True)

    frames_b64 = []

    for fn in frame_numbers:
        # Clamp to valid range
        target = min(fn - 1, total_frames - 1)  # 0-indexed
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()

        if not ret:
            print(f"[CLASSIFIER] WARNING: Could not read frame {fn}, skipping", flush=True)
            continue

        # Encode frame as JPEG -> base64
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode("utf-8")
        frames_b64.append(b64)
        print(f"[CLASSIFIER] Frame {fn} extracted and encoded", flush=True)

    cap.release()
    return frames_b64


# ==================== GROQ CLASSIFICATION ====================
def classify_video(video_path=VIDEO_PATH):
    """
    Main classifier function.
    Returns dict: { "detected": [...], "scenario": "..." }
    Only classes actually present in the video are included.
    """
    print(f"\n{'='*60}", flush=True)
    print("[CLASSIFIER] Starting video pre-classification...", flush=True)
    print(f"[CLASSIFIER] Model: {GROQ_MODEL}", flush=True)
    print(f"[CLASSIFIER] Frames to analyze: {CLASSIFIER_FRAMES}", flush=True)
    print(f"{'='*60}", flush=True)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[CLASSIFIER] ERROR: GROQ_API_KEY is not set in your .env file.", flush=True)
        print("[CLASSIFIER] Cannot classify video without it. Stopping system.", flush=True)
        sys.exit(1)

    # Extract frames
    try:
        frames_b64 = extract_frames(video_path, CLASSIFIER_FRAMES)
    except FileNotFoundError as e:
        print(f"[CLASSIFIER] ERROR: {e}", flush=True)
        sys.exit(1)

    if not frames_b64:
        print("[CLASSIFIER] ERROR: No frames could be extracted from video. Stopping.", flush=True)
        sys.exit(1)

    # Build message with all frames
    content = []
    for i, b64 in enumerate(frames_b64):
        content.append({
            "type": "text",
            "text": f"Frame {CLASSIFIER_FRAMES[i]}:"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })
    content.append({
        "type": "text",
        "text": "Based on these frames, classify what is present. Return ONLY valid JSON."
    })

    # Call Groq
    try:
        client = Groq(api_key=api_key)
        print("[CLASSIFIER] Sending frames to Groq LLM...", flush=True)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": content}
            ],
            max_tokens=200,
            temperature=0.1
        )

        raw = response.choices[0].message.content.strip()
        print(f"[CLASSIFIER] Raw LLM response: {raw}", flush=True)

        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        # Validate structure
        if "detected" not in result:
            result["detected"] = []
        if "scenario" not in result:
            result["scenario"] = "unknown"

        print(f"\n[CLASSIFIER] ✓ Classification complete:", flush=True)
        print(f"[CLASSIFIER]   Detected  : {result['detected']}", flush=True)
        print(f"[CLASSIFIER]   Scenario  : {result['scenario']}", flush=True)
        print(f"{'='*60}\n", flush=True)

        return result

    except json.JSONDecodeError as e:
        print(f"[CLASSIFIER] ERROR: Could not parse Groq response as JSON: {e}", flush=True)
        print("[CLASSIFIER] Stopping system to prevent blind detection.", flush=True)
        sys.exit(1)

    except Exception as e:
        print(f"[CLASSIFIER] ERROR: Groq API call failed: {e}", flush=True)
        print("[CLASSIFIER] Stopping system to prevent blind detection.", flush=True)
        sys.exit(1)


# ==================== FALLBACK ====================
def _fallback_classification():
    """Returns a safe default when classifier fails - agent will handle routing"""
    result = {
        "detected": ["person"],
        "scenario": "Fallback: classifier unavailable, defaulting to person detection"
    }
    print(f"[CLASSIFIER] Using fallback classification: {result}", flush=True)
    return result


# ==================== STANDALONE TEST ====================
if __name__ == "__main__":
    result = classify_video()
    print("Final output:")
    print(json.dumps(result, indent=2))