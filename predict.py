import sys
import os

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
import queue
import json
import threading

from config import (
    VIDEO_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    RESIZE_RATIO, SAVE_OUTPUT_VIDEO, OUTPUT_VIDEO_PATH,
    ALERT_COOLDOWN, MODEL_CONFIGS
)

load_dotenv()


# ==================== POLYGON LOADING ====================
def load_polygon(filename):
    if not filename:
        return None

    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        tracks = data.get("tracks", [])
        if tracks:
            points = tracks[0]['points']
            polygon = np.array(points, np.int32)
            print(f"[DETECTION] Polygon loaded: {len(points)} points")
            return polygon
        else:
            print("[DETECTION] ERROR: No tracks found in polygon file")
            return None

    except FileNotFoundError:
        print(f"[DETECTION] ERROR: Polygon file not found: {filename}")
        return None
    except Exception as e:
        print(f"[DETECTION] ERROR: Error loading polygon: {e}")
        return None


# ==================== SIDE DETECTION ====================
def get_line_side(point, polygon):
    line_x = int(np.mean(polygon[:, 0]))
    return 'right' if point[0] > line_x else 'left'


# ==================== EVENT TRACKER ====================
class DetectionEventTracker:
    def __init__(self):
        self.detection_alert_sent = False
        self.crossing_alerts_sent = 0
        self.max_crossing_alerts = 1

    def should_alert_detection(self, detection_count):
        if detection_count > 0 and not self.detection_alert_sent:
            self.detection_alert_sent = True
            return True
        return False

    def should_alert_crossing(self):
        if self.crossing_alerts_sent < self.max_crossing_alerts:
            self.crossing_alerts_sent += 1
            return True
        return False


# ==================== BOUNDARY TRACKER ====================
class BoundaryTracker:
    def __init__(self, polygon, label):
        self.polygon = polygon
        self.object_positions = {}
        self.last_alert_time = defaultdict(float)
        self.crossing_status = {}
        self.object_id_counter = 0
        self.label = label

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))

    def is_inside_polygon(self, point):
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0

    def update(self, detections):
        current_time = time.time()
        crossed_objects = []

        for bbox, cls_id, cls_name, conf in detections:
            centroid = self.get_centroid(bbox)
            is_inside = self.is_inside_polygon(centroid)
            object_id = self._match_or_create_id(centroid, cls_id)

            if object_id in self.crossing_status:
                prev_status = self.crossing_status[object_id]
                if prev_status != is_inside:
                    if current_time - self.last_alert_time[object_id] > ALERT_COOLDOWN:
                        crossed_objects.append({
                            'id': object_id,
                            'name': cls_name,
                            'centroid': centroid,
                            'direction': 'entering' if is_inside else 'exiting'
                        })
                        self.last_alert_time[object_id] = current_time

            self.crossing_status[object_id] = is_inside
            self.object_positions[object_id] = (centroid, current_time)

        self._cleanup_old_tracks(current_time)
        return crossed_objects

    def _match_or_create_id(self, centroid, cls_id):
        min_distance = 100
        matched_id = None

        for object_id, (prev_centroid, _) in self.object_positions.items():
            distance = np.sqrt(
                (centroid[0] - prev_centroid[0])**2 +
                (centroid[1] - prev_centroid[1])**2
            )
            if distance < min_distance:
                min_distance = distance
                matched_id = object_id

        if matched_id is None:
            matched_id = f"{self.label}_{self.object_id_counter}"
            self.object_id_counter += 1

        return matched_id

    def _cleanup_old_tracks(self, current_time):
        timeout = 5.0
        to_remove = [oid for oid, (_, ts) in self.object_positions.items()
                     if current_time - ts > timeout]
        for oid in to_remove:
            del self.object_positions[oid]
            self.crossing_status.pop(oid, None)


# ==================== MODEL LOADER ====================
def load_models(selected_models):
    loaded = []

    for model_key, cfg in selected_models:
        model_path = cfg["model_path"]
        target_classes = [c.lower() for c in cfg["target_classes"]]

        print(f"[DETECTION] Loading model '{model_key}': {model_path}", flush=True)
        model = YOLO(model_path)
        if torch.cuda.is_available():
            model.to('cuda')

        all_names = model.model.names
        target_ids = {
            cls_id for cls_id, name in all_names.items()
            if name.lower() in target_classes
        }

        if not target_ids:
            print(f"[DETECTION] WARNING: Classes {target_classes} not found in '{model_key}'. Running all classes.", flush=True)
            target_ids = set(all_names.keys())

        print(f"[DETECTION]   → Classes: {[all_names[i] for i in target_ids]}", flush=True)
        loaded.append((model_key, model, target_ids, all_names))

    return loaded


# ==================== MAIN DETECTION ====================
def run_detection(event_queue, selected_models=None, video_path=None, stop_event=None):
    """
    Main detection process.

    Parameters:
        event_queue    : Queue to send detection events to the agent.
        selected_models: List of (model_key, model_config) tuples from agent.
        video_path     : Path to the video file to process. Falls back to config.VIDEO_PATH.
        stop_event     : threading.Event — detection loop exits when this is set.
    """

    # ── Resolve video path ───────────────────────────────────────
    # IMPORTANT: Always use the passed video_path (uploaded file).
    # Only fall back to config VIDEO_PATH if nothing was passed.
    active_video_path = video_path if video_path else VIDEO_PATH
    print(f"[DETECTION] Video path resolved: {active_video_path}", flush=True)

    # ── Resolve models ───────────────────────────────────────────
    if not selected_models:
        print("[DETECTION] WARNING: No models provided, using all from config", flush=True)
        selected_models = list(MODEL_CONFIGS.items())

    # ── Load all YOLO models ─────────────────────────────────────
    loaded_models = load_models(selected_models)
    model_label = " + ".join(key for key, *_ in loaded_models)

    # ── Load polygon boundary from model config ──────────────────
    polygon_file = next(
        (cfg.get("polygon_file") for _, cfg in selected_models if cfg.get("polygon_file")),
        None
    )
    polygon = load_polygon(polygon_file)
    tracker = None

    if polygon is None:
        print("[DETECTION] No boundary defined. Running without crossing alerts.", flush=True)
    else:
        if RESIZE_RATIO != 1.0:
            polygon = (polygon * RESIZE_RATIO).astype(np.int32)
        tracker = BoundaryTracker(polygon, model_label)

    # ── Event tracker ────────────────────────────────────────────
    event_tracker = DetectionEventTracker()

    # ── Open video ───────────────────────────────────────────────
    cap = cv2.VideoCapture(active_video_path)
    if not cap.isOpened():
        print(f"[DETECTION] ERROR: Cannot open video: {active_video_path}")
        event_queue.put(None)
        return

    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(orig_w * RESIZE_RATIO)
    height       = int(orig_h * RESIZE_RATIO)

    # ── Video writer ─────────────────────────────────────────────
    video_writer = None
    if SAVE_OUTPUT_VIDEO:
        os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        if video_writer.isOpened():
            print(f"[DETECTION] Video Writer: Enabled → {OUTPUT_VIDEO_PATH}", flush=True)
        else:
            print(f"[DETECTION] ERROR: Failed to initialize video writer", flush=True)
            video_writer = None

    print(f"\n{'='*70}", flush=True)
    print(f"DETECTION SYSTEM — Models: {model_label.upper()}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Video      : {active_video_path}", flush=True)
    print(f"Resolution : {orig_w}x{orig_h} → {width}x{height} @ {fps} FPS", flush=True)
    print(f"Frames     : {total_frames}", flush=True)
    print(f"Confidence : {CONF_THRESHOLD}", flush=True)
    print(f"Boundary   : {'Yes (' + polygon_file + ')' if polygon is not None else 'No'}", flush=True)
    print(f"{'='*70}\n", flush=True)

    frame_count     = 0
    total_crossings = 0
    total_events    = 0
    LEFT_COLOR      = (0, 255, 0)
    RIGHT_COLOR     = (0, 0, 255)

    try:
        while True:
            # ── Check stop signal ─────────────────────────────────
            if stop_event and stop_event.is_set():
                print("[DETECTION] Stop signal received — halting detection.", flush=True)
                break

            ret, frame = cap.read()
            if not ret:
                break

            if RESIZE_RATIO != 1.0:
                frame = cv2.resize(frame, (width, height))

            frame_count += 1

            # ── Run ALL selected models on this frame ─────────────
            all_detections = []

            for model_key, model, target_ids, all_names in loaded_models:
                results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

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

                        all_detections.append(((x1, y1, x2, y2), cls_id, cls_name, conf))

                        if polygon is not None:
                            centroid = (int((x1 + x2) / 2), y2)
                            side  = get_line_side(centroid, polygon)
                            color = RIGHT_COLOR if side == 'right' else LEFT_COLOR
                        else:
                            color = LEFT_COLOR

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ── Detection alert (once per video) ──────────────────
            if event_tracker.should_alert_detection(len(all_detections)):
                first = all_detections[0]
                event = {
                    'class_name': first[2],
                    'confidence': round(first[3], 2),
                    'count': len(all_detections),
                    'all_classes': list({d[2] for d in all_detections}),
                    'location': 'in_frame',
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'frame': frame_count
                }
                event_queue.put(event)
                time.sleep(0.01)
                total_events += 1
                print(f"[DETECTION] Event sent: {first[2]} detected ({len(all_detections)} objects)", flush=True)

            # ── Boundary crossing ─────────────────────────────────
            if polygon is not None:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], (0, 0, 255))
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                cv2.polylines(frame, [polygon], True, (0, 0, 255), 3)

                if tracker:
                    crossed = tracker.update(all_detections)
                    for obj in crossed:
                        if event_tracker.should_alert_crossing():
                            event = {
                                'class_name': obj['name'],
                                'confidence': round(all_detections[0][3], 2) if all_detections else 0.0,
                                'count': 1,
                                'location': 'boundary_zone',
                                'direction': obj['direction'],
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'frame': frame_count
                            }
                            event_queue.put(event)
                            time.sleep(0.01)
                            total_crossings += 1
                            total_events += 1
                            print(f"[DETECTION] Event sent: {obj['name']} boundary crossing ({obj['direction']})", flush=True)

                        cv2.putText(frame, f"!!! {obj['name'].upper()} CROSSED BOUNDARY !!!",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # ── HUD ───────────────────────────────────────────────
            info = f"Frame: {frame_count}/{total_frames} | Detected: {len(all_detections)} | Events: {total_events}"
            if polygon is not None:
                info += f" | Crossings: {total_crossings}"
            cv2.putText(frame, info, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if video_writer is not None:
                video_writer.write(frame)

            cv2.imshow(f"Detection: {model_label}", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"\n[DETECTION] Output saved: {OUTPUT_VIDEO_PATH}")
        cv2.destroyAllWindows()

        event_queue.put(None)  # Shutdown signal to agent

        stopped_early = stop_event and stop_event.is_set()
        print(f"\n{'='*70}")
        print("DETECTION COMPLETE" + (" (Stopped by user)" if stopped_early else ""))
        print(f"{'='*70}")
        print(f"[DETECTION] Models used      : {model_label}")
        print(f"[DETECTION] Frames processed : {frame_count}/{total_frames}")
        print(f"[DETECTION] Total events     : {total_events}")
        if polygon is not None:
            print(f"[DETECTION] Total crossings  : {total_crossings}")
        print(f"{'='*70}")

# ==================== STANDALONE ====================
if __name__ == "__main__":
    print("[INFO] Running detection in standalone mode (no alerts)")
    print("[INFO] To enable alerts, use launcher.py\n")
    dummy_queue = queue.Queue()
    run_detection(dummy_queue)