# ==================== RTSP_HANDLERS.PY ====================
# Manages per-camera RTSP detection state.
# Each camera gets its own thread running YOLO on the RTSP stream.
# Annotated frames are stored in a shared state dict and served as MJPEG.

import cv2
import time
import threading
import logging
import numpy as np
from collections import defaultdict

from predict import load_models, load_polygon, BoundaryTracker, DetectionEventTracker, get_line_side
from config import (
    MODEL_CONFIGS, CONF_THRESHOLD, IOU_THRESHOLD,
    RESIZE_RATIO, ALERT_COOLDOWN, RTSP_CAMERAS, RTSP_CONNECT_TIMEOUT
)

logger = logging.getLogger(__name__)

# ── Global per-camera state dict ─────────────────────────────────────────────
RTSP_STATES: dict = {}


def _make_state(camera_id: str) -> dict:
    return {
        "camera_id":   camera_id,
        "is_running":  False,
        "stop_flag":   threading.Event(),
        "frame":       None,          # latest JPEG bytes
        "lock":        threading.Lock(),
        "thread":      None,
        "model_keys":  [],
        "error":       None,
        "fps":         0.0,
        "detections":  0,
        "frame_count": 0,
    }


def get_state(camera_id: str) -> dict:
    if camera_id not in RTSP_STATES:
        RTSP_STATES[camera_id] = _make_state(camera_id)
    return RTSP_STATES[camera_id]


# ==================== PUBLIC API ====================

def start_rtsp_detection(camera_id: str, model_keys: list) -> tuple:
    """
    Start YOLO detection on the RTSP stream for camera_id.
    Returns (success: bool, message: str).
    """
    rtsp_url = RTSP_CAMERAS.get(camera_id, "")
    if not rtsp_url:
        return False, f"No RTSP URL configured for {camera_id}"

    if not model_keys:
        return False, "No models selected"

    invalid = [k for k in model_keys if k not in MODEL_CONFIGS]
    if invalid:
        return False, f"Unknown model keys: {invalid}"

    state = get_state(camera_id)

    if state["is_running"]:
        return False, f"{camera_id} is already running"

    state["stop_flag"].clear()
    state["is_running"]  = True
    state["error"]       = None
    state["model_keys"]  = model_keys
    state["frame"]       = None
    state["detections"]  = 0
    state["frame_count"] = 0

    thread = threading.Thread(
        target=_rtsp_worker,
        args=(camera_id, rtsp_url, model_keys, state),
        daemon=True,
        name=f"rtsp-{camera_id}"
    )
    thread.start()
    state["thread"] = thread

    logger.info(f"[RTSP] Started detection on {camera_id} → models: {model_keys}")
    return True, "Detection started"


def stop_rtsp_detection(camera_id: str) -> tuple:
    """Signal a running RTSP detection to stop."""
    state = RTSP_STATES.get(camera_id)
    if not state or not state["is_running"]:
        return False, f"{camera_id} is not running"

    state["stop_flag"].set()
    logger.info(f"[RTSP] Stop signal sent to {camera_id}")
    return True, "Stop signal sent"


def get_rtsp_status(camera_id: str) -> dict:
    state = RTSP_STATES.get(camera_id)
    rtsp_url = RTSP_CAMERAS.get(camera_id, "")
    if not state:
        return {
            "camera_id":   camera_id,
            "is_running":  False,
            "model_keys":  [],
            "detections":  0,
            "frame_count": 0,
            "fps":         0.0,
            "error":       None,
            "configured":  bool(rtsp_url),
        }
    return {
        "camera_id":   camera_id,
        "is_running":  state["is_running"],
        "model_keys":  state["model_keys"],
        "detections":  state["detections"],
        "frame_count": state["frame_count"],
        "fps":         round(state["fps"], 1),
        "error":       state["error"],
        "configured":  bool(rtsp_url),
    }


def rtsp_mjpeg_streamer(camera_id: str):
    """
    Sync generator — yields MJPEG boundary frames from the camera's state buffer.
    Used directly in FastAPI StreamingResponse.
    """
    state = RTSP_STATES.get(camera_id)
    if not state:
        return

    no_frame_count = 0
    max_no_frame   = 60   # ~3s at 20fps polling

    while True:
        if not state["is_running"] and no_frame_count >= max_no_frame:
            logger.info(f"[RTSP] MJPEG stream ended for {camera_id}")
            break

        with state["lock"]:
            frame = state["frame"]

        if frame is not None:
            no_frame_count = 0
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        else:
            no_frame_count += 1
            time.sleep(0.05)
            continue

        time.sleep(0.033)   # ~30fps cap to client


# ==================== INTERNAL WORKER ====================

def _rtsp_worker(camera_id: str, rtsp_url: str, model_keys: list, state: dict):
    """
    Runs in its own daemon thread.
    Opens RTSP stream, runs selected YOLO models on each frame,
    encodes annotated frames as JPEG and writes to state['frame'].
    """
    logger.info(f"[RTSP] Worker starting: {camera_id} → {rtsp_url}")
    cap = None

    try:
        # ── Load YOLO models ─────────────────────────────────────
        selected_models = [(k, MODEL_CONFIGS[k]) for k in model_keys if k in MODEL_CONFIGS]
        loaded_models   = load_models(selected_models)
        model_label     = " + ".join(k for k, *_ in loaded_models)

        # ── Load per-model polygon + tracker ─────────────────────
        # Each model gets its own polygon and boundary tracker
        model_polygons = {}   # model_key → np.array or None
        model_trackers = {}   # model_key → BoundaryTracker or None
        for model_key, cfg in selected_models:
            pfile   = cfg.get("polygon_file")
            polygon = load_polygon(pfile)
            if polygon is not None:
                if RESIZE_RATIO != 1.0:
                    polygon = (polygon * RESIZE_RATIO).astype(np.int32)
                model_polygons[model_key] = polygon
                model_trackers[model_key] = BoundaryTracker(polygon, model_key)
                logger.info(f"[RTSP] {camera_id}: Polygon loaded for '{model_key}'")
            else:
                model_polygons[model_key] = None
                model_trackers[model_key] = None

        # ── Open RTSP stream ─────────────────────────────────────
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_CONNECT_TIMEOUT * 1000)

        if not cap.isOpened():
            state["error"]      = f"Cannot open stream: {rtsp_url}"
            state["is_running"] = False
            logger.error(f"[RTSP] {camera_id}: {state['error']}")
            return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width  = int(orig_w * RESIZE_RATIO)
        height = int(orig_h * RESIZE_RATIO)

        logger.info(f"[RTSP] {camera_id}: Stream opened {orig_w}x{orig_h} → {width}x{height}")

        LEFT_COLOR  = (0, 255, 0)
        RIGHT_COLOR = (0, 0, 255)

        fps_start  = time.time()
        fps_frames = 0
        consecutive_fails = 0
        MAX_CONSEC_FAILS  = 10  # skip up to 10 bad/corrupted frames before reconnecting

        # ── Frame detection loop ─────────────────────────────────
        while not state["stop_flag"].is_set():
            ret, frame = cap.read()
            
            if not ret:
                consecutive_fails += 1
                if consecutive_fails < MAX_CONSEC_FAILS:
                    # Likely a corrupted H264 packet — skip and keep going
                    time.sleep(0.02)
                    continue
                # Too many consecutive failures — real disconnect, reconnect
                logger.warning(f"[RTSP] {camera_id}: {consecutive_fails} consecutive frame failures — reconnecting")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    state["error"] = "Stream dropped and reconnect failed"
                    break
                consecutive_fails = 0
                continue

            consecutive_fails = 0 # reset on any good frame

            if RESIZE_RATIO != 1.0:
                frame = cv2.resize(frame, (width, height))

            state["frame_count"] += 1
            fps_frames += 1

            # ── Run YOLO models ───────────────────────────────────
            all_detections = []

            for model_key, model, target_ids, all_names in loaded_models:
                polygon = model_polygons.get(model_key)
                tracker = model_trackers.get(model_key)

                results = model.predict(frame, conf=CONF_THRESHOLD,
                                        iou=IOU_THRESHOLD, verbose=False)
                model_dets = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for i in range(len(r.boxes)):
                        cls_id = int(r.boxes.cls[i])
                        if cls_id not in target_ids:
                            continue

                        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                        conf     = float(r.boxes.conf[i])
                        cls_name = all_names[cls_id]
                        det      = ((x1, y1, x2, y2), cls_id, cls_name, conf)
                        model_dets.append(det)
                        all_detections.append(det)

                        if polygon is not None:
                            centroid = (int((x1 + x2) / 2), y2)
                            side     = get_line_side(centroid, polygon)
                            color    = RIGHT_COLOR if side == "right" else LEFT_COLOR
                        else:
                            color = LEFT_COLOR

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{cls_name} {conf:.2f}",
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, color, 2)

                # ── Per-model boundary overlay + crossing alerts ──
                if polygon is not None:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [polygon], (0, 0, 255))
                    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                    cv2.polylines(frame, [polygon], True, (0, 0, 255), 3)

                    if tracker:
                        crossed = tracker.update(model_dets)
                        for obj in crossed:
                            cv2.putText(frame,
                                        f"!!! {obj['name'].upper()} CROSSED [{model_key}] !!!",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 0, 255), 3)

            state["detections"] = len(all_detections)

            # ── HUD ───────────────────────────────────────────────
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                state["fps"] = fps_frames / elapsed
                fps_frames   = 0
                fps_start    = time.time()

            hud = (f"[{camera_id.upper()}] {model_label} | "
                   f"Det: {len(all_detections)} | "
                   f"FPS: {state['fps']:.1f}")
            cv2.putText(frame, hud,
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            ts = time.strftime("%H:%M:%S")
            cv2.putText(frame, ts,
                        (frame.shape[1] - 75, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

            # ── Store encoded frame ───────────────────────────────
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state["lock"]:
                state["frame"] = buf.tobytes()

    except Exception as e:
        state["error"] = str(e)
        logger.error(f"[RTSP] {camera_id} worker error: {e}", exc_info=True)

    finally:
        if cap:
            cap.release()
        state["is_running"] = False
        logger.info(f"[RTSP] Worker stopped: {camera_id} | "
                    f"frames={state['frame_count']} error={state['error']}")