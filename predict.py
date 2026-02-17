import sys
import os

# Fix Windows encoding issues AND force line buffering
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np
from collections import defaultdict 
from dotenv import load_dotenv
import queue

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
# Model Configuration
MODEL_PATH = "weights/fire_l.pt"
TARGET_CLASS_ID = 1

# Input Configuration
VIDEO_PATH = "input_videos/fire_video1.mp4"
POLYGON_FILE = ""

# Detection Parameters
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

# Video Resize
RESIZE_RATIO = 0.5

# Color Configuration
LEFT_SIDE_COLOR = (0, 255, 0)
RIGHT_SIDE_COLOR = (0, 0, 255)

# Video Saving
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = "output_videos/detection_output.mp4"

# Alert Configuration
ALERT_COOLDOWN = 60


# ==================== POLYGON LOADING ====================
def load_polygon(filename):
    """Load polygon boundary from JSON file"""
    if not filename:
        return None
        
    try:
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        
        tracks = data.get("tracks", [])
        if tracks:
            points = tracks[0]['points']
            polygon = np.array(points, np.int32)
            print(f"[DETECTION] Polygon loaded: {len(points)} points")
            return polygon
        else:
            print("[DETECTION] ERROR: No tracks found in file")
            return None
    except FileNotFoundError:
        print(f"[DETECTION] ERROR: Polygon file not found: {filename}")
        return None
    except Exception as e:
        print(f"[DETECTION] ERROR: Error loading polygon: {e}")
        return None


# ==================== SIDE DETECTION ====================
def get_line_side(point, polygon):
    """Determine which side of the vertical line a point is on"""
    line_x = int(np.mean(polygon[:, 0]))
    return 'right' if point[0] > line_x else 'left'


# ==================== EVENT DETECTION ====================
class DetectionEventTracker:
    """Track detection events - maximum 2 events per video (1 detection + 1 crossing)"""
    
    def __init__(self):
        self.detection_alert_sent = False
        self.crossing_alerts_sent = 0
        self.max_crossing_alerts = 1
    
    def should_alert_detection(self, detection_count):
        """Check if detection alert should be triggered (once per video)"""
        if detection_count > 0 and not self.detection_alert_sent:
            self.detection_alert_sent = True
            return True
        return False
    
    def should_alert_crossing(self):
        """Check if crossing alert should be triggered (once per video)"""
        if self.crossing_alerts_sent < self.max_crossing_alerts:
            self.crossing_alerts_sent += 1
            return True
        return False


# ==================== CROSSING DETECTION ====================
class BoundaryTracker:
    """Track objects and detect boundary crossings"""
    
    def __init__(self, polygon, model_name):
        self.polygon = polygon
        self.object_positions = {}
        self.last_alert_time = defaultdict(float)
        self.crossing_status = {}
        self.object_id_counter = 0
        self.model_name = model_name
        
    def get_centroid(self, bbox):
        """Get bottom-middle point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))
    
    def is_inside_polygon(self, point):
        """Check if point is inside polygon"""
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0
    
    def update(self, detections):
        """Update tracker with new detections"""
        current_time = time.time()
        crossed_objects = []
        
        for bbox, cls_id, cls_name, conf in detections:
            centroid = self.get_centroid(bbox)
            is_inside = self.is_inside_polygon(centroid)
            
            object_id = self._match_or_create_id(centroid, cls_id)
            
            if object_id in self.crossing_status:
                previous_status = self.crossing_status[object_id]
                
                if previous_status != is_inside:
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
        """Match centroid to existing object or create new ID"""
        min_distance = 100
        matched_id = None
        
        for object_id, (prev_centroid, _) in self.object_positions.items():
            distance = np.sqrt((centroid[0] - prev_centroid[0])**2 + 
                             (centroid[1] - prev_centroid[1])**2)
            if distance < min_distance:
                min_distance = distance
                matched_id = object_id
        
        if matched_id is None:
            matched_id = f"{self.model_name}_{self.object_id_counter}"
            self.object_id_counter += 1
        
        return matched_id
    
    def _cleanup_old_tracks(self, current_time):
        """Remove objects not detected for 5 seconds"""
        timeout = 5.0
        to_remove = []
        
        for object_id, (_, timestamp) in self.object_positions.items():
            if current_time - timestamp > timeout:
                to_remove.append(object_id)
        
        for object_id in to_remove:
            del self.object_positions[object_id]
            if object_id in self.crossing_status:
                del self.crossing_status[object_id]


# ==================== DETECTION PROCESS ====================
def run_detection(event_queue):
    """Main detection process - sends events to queue for agent processing"""
    
    # Load model
    model = YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        model.to('cuda')
    
    print(f"[DETECTION] Model loaded: {MODEL_PATH}", flush=True)
    print(f"[DETECTION] Target Class: {model.model.names[TARGET_CLASS_ID]} (class {TARGET_CLASS_ID})", flush=True)
    print(f"[DETECTION] All classes: {model.model.names}", flush=True)
    
    # Load polygon
    polygon = load_polygon(POLYGON_FILE)
    if polygon is None:
        print("\n[DETECTION] No boundary defined. Running without crossing alerts.")
        tracker = None
    else:
        if RESIZE_RATIO != 1.0:
            polygon = (polygon * RESIZE_RATIO).astype(np.int32)
        tracker = BoundaryTracker(polygon, model.model.names[TARGET_CLASS_ID])
    
    # Initialize event tracker
    event_tracker = DetectionEventTracker()
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[DETECTION] ERROR: Cannot open video: {VIDEO_PATH}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width = int(original_width * RESIZE_RATIO)
    height = int(original_height * RESIZE_RATIO)
    
    # Setup video writer
    video_writer = None
    if SAVE_OUTPUT_VIDEO:
        os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        if video_writer.isOpened():
            print(f"[DETECTION] Video Writer: Enabled")
            print(f"[DETECTION] Output: {OUTPUT_VIDEO_PATH}")
        else:
            print(f"[DETECTION] ERROR: Failed to initialize video writer")
            video_writer = None
    
    print(f"\n{'='*70}", flush=True)
    print(f"{model.model.names[TARGET_CLASS_ID].upper()} DETECTION SYSTEM", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Video: {VIDEO_PATH}", flush=True)
    if RESIZE_RATIO != 1.0:
        print(f"Resolution: {original_width}x{original_height} -> {width}x{height} @ {fps} FPS", flush=True)
    else:
        print(f"Resolution: {width}x{height} @ {fps} FPS", flush=True)
    print(f"Total Frames: {total_frames}", flush=True)
    print(f"Confidence: {CONF_THRESHOLD}", flush=True)
    print(f"\nPress 'q' to quit", flush=True)
    print(f"{'='*70}\n", flush=True)
    print("[DETECTION] 🎬 Opening video window NOW...\n", flush=True)
    
    frame_count = 0
    total_crossings = 0
    total_events = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if RESIZE_RATIO != 1.0:
                frame = cv2.resize(frame, (width, height))
            
            frame_count += 1
            
            # Run detection
            results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            detections = []
            
            for r in results:
                if r.boxes is not None:
                    for i in range(len(r.boxes)):
                        cls_id = int(r.boxes.cls[i])
                        
                        if cls_id != TARGET_CLASS_ID:
                            continue
                        
                        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                        conf = float(r.boxes.conf[i])
                        cls_name = model.model.names[cls_id]
                        
                        detections.append(((x1, y1, x2, y2), cls_id, cls_name, conf))
                        
                        if polygon is not None:
                            centroid = (int((x1 + x2) / 2), y2)
                            side = get_line_side(centroid, polygon)
                            color = RIGHT_SIDE_COLOR if side == 'right' else LEFT_SIDE_COLOR
                        else:
                            color = LEFT_SIDE_COLOR
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Send detection event to queue (once per video)
            if event_tracker.should_alert_detection(len(detections)):
                # Send raw facts - agent decides what to do
                event = {
                    'class_name': model.model.names[TARGET_CLASS_ID],
                    'confidence': round(float(detections[0][3]), 2) if detections else 0.0,
                    'count': len(detections),
                    'location': 'in_frame',
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'frame': frame_count
                }
                event_queue.put(event)
                time.sleep(0.01)
                total_events += 1
                print(f"[DETECTION] Event sent: {model.model.names[TARGET_CLASS_ID]} detected ({len(detections)} objects)", flush=True)
            
            # Handle boundary crossing
            if polygon is not None:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], (0, 0, 255))
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                cv2.polylines(frame, [polygon], True, (0, 0, 255), 3)
                
                if tracker:
                    crossed = tracker.update(detections)
                    
                    for obj in crossed:
                        if event_tracker.should_alert_crossing():
                            # Send raw facts - agent decides what to do
                            event = {
                                'class_name': obj['name'],
                                'confidence': round(float(detections[0][3]), 2) if detections else 0.0,
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
            
            # Display info
            info_text = f"Frame: {frame_count}/{total_frames} | {model.model.names[TARGET_CLASS_ID].title()}: {len(detections)} | Events: {total_events}"
            if polygon is not None:
                info_text += f" | Crossings: {total_crossings}"
            
            cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if video_writer is not None:
                video_writer.write(frame)
            
            cv2.imshow(f"{model.model.names[TARGET_CLASS_ID].title()} Detection", frame)
            
            # 10ms wait for smooth playback while allowing thread switching
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"\n[DETECTION] Output video saved: {OUTPUT_VIDEO_PATH}")
        cv2.destroyAllWindows()
        
        # Signal agent to stop
        event_queue.put(None)
        
        print(f"\n{'='*70}")
        print(f"DETECTION COMPLETE")
        print(f"{'='*70}")
        print(f"[DETECTION] Frames processed: {frame_count}/{total_frames}")
        print(f"[DETECTION] Total events sent: {total_events}")
        if polygon is not None:
            print(f"[DETECTION] Total crossings: {total_crossings}")
        print(f"{'='*70}")


# ==================== STANDALONE MODE ====================
if __name__ == "__main__":
    # Run in standalone mode (no agent)
    print("[INFO] Running detection in standalone mode (no alerts)")
    print("[INFO] To enable alerts, use launcher.py\n")
    
    # Create dummy queue
    dummy_queue = queue.Queue()
    run_detection(dummy_queue)