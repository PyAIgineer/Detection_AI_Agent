# ==================== JEWELLERY_CLASSIFIER.PY ====================
# Extracts frames from input video and sends them to Groq vision LLM
# to identify jewellery items before running YOLO.

import sys
import cv2
import json
import base64
from groq import Groq
from dotenv import load_dotenv
import os

from config import VIDEO_PATH, CLASSIFIER_FRAMES, GROQ_MODEL

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ==================== PROMPT ====================
JEWELLERY_CLASSIFIER_PROMPT = """You are a jewellery detection classifier. You will receive 3 frames from a surveillance video.

Your job is to identify ONLY jewellery items that are CLEARLY VISIBLE in these frames.

Detectable categories:
- necklace
- earrings

Rules:
- Only include classes you can clearly see. Do NOT guess.
- If neither is visible, return an empty detected list.

Respond ONLY in this JSON format with no extra text:
{
    "detected": ["class1", "class2"],
    "scenario": "brief one-line description of jewellery visible in the video"
}"""


# ==================== FRAME EXTRACTION ====================
def extract_frames(video_path, frame_numbers):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[JEWELLERY-CLASSIFIER] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_b64 = []

    for fn in frame_numbers:
        target = min(fn - 1, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode("utf-8")
        frames_b64.append(b64)

    cap.release()
    return frames_b64


# ==================== GROQ CLASSIFICATION ====================
def classify_jewellery(video_path=VIDEO_PATH):
    """
    Runs jewellery classification on the video.
    Returns dict: { "detected": [...], "scenario": "..." }
    """
    print(f"\n{'='*60}", flush=True)
    print("[JEWELLERY-CLASSIFIER] Starting jewellery pre-classification...", flush=True)
    print(f"[JEWELLERY-CLASSIFIER] Model: {GROQ_MODEL}", flush=True)
    print(f"[JEWELLERY-CLASSIFIER] Frames to analyze: {CLASSIFIER_FRAMES}", flush=True)
    print(f"{'='*60}", flush=True)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[JEWELLERY-CLASSIFIER] ERROR: GROQ_API_KEY not set. Skipping jewellery classification.", flush=True)
        return {"detected": [], "scenario": "unavailable"}

    try:
        frames_b64 = extract_frames(video_path, CLASSIFIER_FRAMES)
    except FileNotFoundError as e:
        print(f"[JEWELLERY-CLASSIFIER] ERROR: {e}", flush=True)
        return {"detected": [], "scenario": "unavailable"}

    if not frames_b64:
        print("[JEWELLERY-CLASSIFIER] No frames extracted. Skipping.", flush=True)
        return {"detected": [], "scenario": "unavailable"}

    content = []
    for i, b64 in enumerate(frames_b64):
        content.append({"type": "text", "text": f"Frame {CLASSIFIER_FRAMES[i]}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": "Based on these frames, classify jewellery present. Return ONLY valid JSON."})

    client = Groq(api_key=api_key)
    print("[JEWELLERY-CLASSIFIER] Sending frames to Groq LLM...", flush=True)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": JEWELLERY_CLASSIFIER_PROMPT},
            {"role": "user", "content": content}
        ],
        max_tokens=150,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    print(f"[JEWELLERY-CLASSIFIER] Raw LLM response: {raw}", flush=True)

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    if "detected" not in result:
        result["detected"] = []
    if "scenario" not in result:
        result["scenario"] = "unknown"

    print(f"[JEWELLERY-CLASSIFIER] ✓ Classification complete:", flush=True)
    print(f"[JEWELLERY-CLASSIFIER]   Detected : {result['detected']}", flush=True)
    print(f"[JEWELLERY-CLASSIFIER]   Scenario : {result['scenario']}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return result


if __name__ == "__main__":
    result = classify_jewellery()
    print(json.dumps(result, indent=2))