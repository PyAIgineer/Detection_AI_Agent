# ==================== APP.PY - FastAPI Backend ====================

import os
import json
import uuid
import asyncio
import threading
import queue
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from video_classifier import classify_video
from prediction_agent import AlertAgent
from predict import run_detection
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH, MODEL_CONFIGS, RTSP_CAMERAS
from rtsp_handlers import (
    start_rtsp_detection, stop_rtsp_detection,
    get_rtsp_status, rtsp_mjpeg_streamer
)

# ==================== SETUP ====================
app = FastAPI(title="AI Detection Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("output_videos")
LOGS_DIR   = Path("logs")
DB_FILE    = Path("detection_history.json")

for d in [OUTPUT_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

tasks      = {}
stop_flags = {}

# ==================== PYDANTIC MODELS ====================
class PredictionTask:
    def __init__(self, task_id: str, video_path: str):
        self.task_id           = task_id
        self.video_path        = video_path
        self.status            = "pending"
        self.classifier_output = None
        self.selected_models   = []
        self.logs              = []
        self.results           = {}
        self.output_video      = None
        self.started_at        = None
        self.completed_at      = None
        self.error             = None

class TaskStatus(BaseModel):
    task_id:            str
    status:             str
    classifier_output:  Optional[dict] = None
    selected_models:    Optional[list] = None
    results:            Optional[dict] = None
    output_video:       Optional[str]  = None
    error:              Optional[str]  = None

class PredictRequest(BaseModel):
    video_path: str

class RTSPStartRequest(BaseModel):
    model_keys: List[str]   # e.g. ["deer", "fire_smoke"]

# ==================== LOG CAPTURE ====================
class LogCapture:
    def __init__(self, task: PredictionTask):
        self.task            = task
        self.original_stdout = None
        self.original_stderr = None
        self._buffer         = ""

    def write(self, text):
        if self.original_stdout:
            self.original_stdout.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.task.logs.append(f"[{timestamp}]  {line}")

    def flush(self):
        if self._buffer.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.task.logs.append(f"[{timestamp}]  {self._buffer.strip()}")
            self._buffer = ""
        if self.original_stdout:
            self.original_stdout.flush()

# ==================== DETECTION WORKER ====================
def run_detection_task(task: PredictionTask, stop_event: threading.Event):
    import sys
    log_capture = LogCapture(task)

    try:
        task.status     = "running"
        task.started_at = datetime.now().isoformat()

        log_capture.original_stdout = sys.stdout
        log_capture.original_stderr = sys.stderr
        sys.stdout = log_capture
        sys.stderr = log_capture

        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ── STEP 1/3: Video Classification ──")
        classifier_output = classify_video(task.video_path)
        task.classifier_output = classifier_output
        task.logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}]  ✔ Classifier done → {classifier_output['detected']}"
        )

        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ── STEP 2/3: Model Routing ──")
        agent = AlertAgent()
        selected_models = agent.decide_models(classifier_output)
        task.selected_models = [key for key, _ in selected_models]
        task.logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}]  ✔ Models selected: {task.selected_models}"
        )

        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ── STEP 3/3: Detection ──")

        event_queue = queue.Queue()
        agent_thread = threading.Thread(target=agent.run, args=(event_queue,), daemon=True)
        agent_thread.start()

        output_path = OUTPUT_DIR / f"{task.task_id}_output.mp4"
        run_detection(
            event_queue,
            selected_models,
            video_path=task.video_path,
            stop_event=stop_event
        )
        task.output_video = str(output_path)

        if stop_event.is_set():
            task.status = "stopped"
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ⏹ Detection stopped by user")
        else:
            task.status = "completed"
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ✔ Detection complete")

        task.results = {
            "video_processed": True,
            "models_used":     task.selected_models,
            "output_video":    str(output_path),
            "timestamp":       datetime.now().isoformat()
        }
        task.completed_at = datetime.now().isoformat()
        save_to_history(task)

    except Exception as e:
        task.status       = "failed"
        task.error        = str(e)
        task.completed_at = datetime.now().isoformat()
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ✖ ERROR: {str(e)}")

    finally:
        sys.stdout = log_capture.original_stdout
        sys.stderr = log_capture.original_stderr
        try:
            if task.video_path and os.path.exists(task.video_path):
                os.remove(task.video_path)
        except Exception:
            pass

# ==================== HISTORY ====================
def save_to_history(task: PredictionTask):
    history = []
    if DB_FILE.exists():
        with open(DB_FILE, 'r') as f:
            history = json.load(f)
    history.append({
        "task_id":           task.task_id,
        "video_name":        Path(task.video_path).name,
        "status":            task.status,
        "classifier_output": task.classifier_output,
        "selected_models":   task.selected_models,
        "results":           task.results,
        "started_at":        task.started_at,
        "completed_at":      task.completed_at,
        "error":             task.error
    })
    with open(DB_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_history():
    if DB_FILE.exists():
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return []

# ==================== VIDEO UPLOAD & DETECT ====================
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(400, "Only video files are allowed")

    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
    finally:
        tmp.close()
    return {"message": "Video uploaded", "path": tmp.name, "original_name": file.filename}

@app.post("/predict")
async def start_prediction(background_tasks: BackgroundTasks, request: PredictRequest):
    if not Path(request.video_path).exists():
        raise HTTPException(404, "Video file not found — please upload again")

    task_id    = str(uuid.uuid4())[:8]
    task       = PredictionTask(task_id, request.video_path)
    stop_event = threading.Event()

    tasks[task_id]      = task
    stop_flags[task_id] = stop_event

    background_tasks.add_task(run_detection_task, task, stop_event)
    return {"task_id": task_id, "status": "started"}

@app.post("/stop/{task_id}")
async def stop_detection(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    if task.status not in ("running", "pending"):
        return {"message": f"Task is already {task.status}", "task_id": task_id}
    stop_event = stop_flags.get(task_id)
    if stop_event:
        stop_event.set()
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ⏹ Stop requested by user")
    return {"message": "Stop signal sent", "task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    return TaskStatus(
        task_id=task.task_id, status=task.status,
        classifier_output=task.classifier_output, selected_models=task.selected_models,
        results=task.results, output_video=task.output_video, error=task.error
    )

@app.get("/logs/{task_id}")
async def stream_logs(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]

    async def event_stream():
        last_index = 0
        while True:
            if last_index < len(task.logs):
                for log in task.logs[last_index:]:
                    yield f"data: {json.dumps({'log': log})}\n\n"
                last_index = len(task.logs)
            if task.status in ("completed", "failed", "stopped"):
                yield f"data: {json.dumps({'status': task.status, 'done': True})}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/download/{task_id}")
async def download_output(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    if not task.output_video or not Path(task.output_video).exists():
        raise HTTPException(404, "Output video not found")
    return FileResponse(task.output_video, media_type="video/mp4",
                        filename=f"detection_{task_id}.mp4")

# ==================== RTSP CAMERA ENDPOINTS ====================

@app.get("/rtsp/cameras")
async def get_cameras():
    """All camera IDs from config with RTSP URL presence and live detection status."""
    cameras = []
    for camera_id, rtsp_url in RTSP_CAMERAS.items():
        status = get_rtsp_status(camera_id)
        cameras.append({
            "camera_id":  camera_id,
            "label":      camera_id.upper().replace("_", " "),
            "configured": bool(rtsp_url),
            "is_running": status["is_running"],
            "model_keys": status["model_keys"],
            "detections": status["detections"],
            "fps":        status["fps"],
            "error":      status["error"],
        })
    return cameras


@app.get("/rtsp/models")
async def get_available_models():
    """All model keys and their target classes from MODEL_CONFIGS."""
    return {
        key: {
            "target_classes": cfg["target_classes"],
            "has_boundary":   bool(cfg.get("polygon_file")),
        }
        for key, cfg in MODEL_CONFIGS.items()
    }


@app.post("/rtsp/start/{camera_id}")
async def rtsp_start(camera_id: str, request: RTSPStartRequest):
    """Start YOLO detection on an RTSP camera with the selected model_keys."""
    if camera_id not in RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found in config")

    success, message = start_rtsp_detection(camera_id, request.model_keys)
    if not success:
        raise HTTPException(400, message)

    return {"camera_id": camera_id, "message": message, "model_keys": request.model_keys}


@app.post("/rtsp/stop/{camera_id}")
async def rtsp_stop(camera_id: str):
    """Stop detection on an RTSP camera."""
    if camera_id not in RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found in config")

    success, message = stop_rtsp_detection(camera_id)
    return {"camera_id": camera_id, "message": message}


@app.get("/rtsp/status/{camera_id}")
async def rtsp_status_endpoint(camera_id: str):
    """Live detection status for one camera."""
    if camera_id not in RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found in config")
    return get_rtsp_status(camera_id)


@app.get("/rtsp/stream/{camera_id}")
async def rtsp_stream_endpoint(camera_id: str):
    """
    MJPEG stream of annotated RTSP frames.
    Use as <img src="/rtsp/stream/cctv_01"> — browser updates automatically.
    """
    if camera_id not in RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found in config")

    status = get_rtsp_status(camera_id)
    if not status["is_running"]:
        raise HTTPException(409, f"{camera_id} is not running. Start detection first.")

    async def async_stream():
        """Wrap the sync MJPEG generator so it doesn't block the event loop."""
        loop = asyncio.get_event_loop()
        gen  = rtsp_mjpeg_streamer(camera_id)
        while True:
            chunk = await loop.run_in_executor(None, next, gen, None)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        async_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ==================== ANALYTICS & HISTORY ====================
@app.get("/analytics")
async def get_analytics():
    history       = load_history()
    status_counts = {"completed": 0, "failed": 0, "stopped": 0}
    model_usage   = {}
    detections_over_time = []

    for record in history:
        status = record.get("status", "unknown")
        if status in status_counts:
            status_counts[status] += 1
        for model in record.get("selected_models", []):
            model_usage[model] = model_usage.get(model, 0) + 1
        if record.get("completed_at"):
            detections_over_time.append({
                "timestamp": record["completed_at"],
                "models":    record.get("selected_models", []),
                "detected":  record.get("classifier_output", {}).get("detected", [])
            })

    return {
        "total_detections":     len(history),
        "status_counts":        status_counts,
        "model_usage":          model_usage,
        "recent_detections":    history[-10:],
        "detections_over_time": detections_over_time[-20:]
    }

@app.get("/history")
async def get_history():
    return load_history()

@app.delete("/history")
async def clear_history():
    if DB_FILE.exists():
        DB_FILE.unlink()
    return {"message": "History cleared"}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import logging

    class FilterNoise(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "GET /analytics" not in msg and "GET /rtsp/status" not in msg

    logging.getLogger("uvicorn.access").addFilter(FilterNoise())

    print("=" * 70)
    print("AI DETECTION DASHBOARD")
    print("=" * 70)
    print("http://localhost:8000")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)