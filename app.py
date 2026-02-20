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
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from video_classifier import classify_video
from prediction_agent import AlertAgent
from predict import run_detection
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH, MODEL_CONFIGS

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

# In-memory task storage
tasks      = {}
stop_flags = {}   # task_id → threading.Event

# ==================== MODELS ====================
class PredictionTask:
    def __init__(self, task_id: str, video_path: str):
        self.task_id        = task_id
        self.video_path     = video_path   # actual path of the uploaded temp file
        self.status         = "pending"    # pending | running | completed | failed | stopped
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

# ==================== HELPER: LOG CAPTURE ====================
class LogCapture:
    """
    Redirects stdout/stderr to task.logs.
    Handles partial writes (Python's print calls write() twice: text + '\\n').
    Each non-empty line becomes one timestamped log entry.
    """
    def __init__(self, task: PredictionTask):
        self.task            = task
        self.original_stdout = None
        self.original_stderr = None
        self._buffer         = ""          # accumulate partial writes

    def write(self, text):
        # Always mirror to real terminal
        if self.original_stdout:
            self.original_stdout.write(text)

        self._buffer += text

        # Flush complete lines to task.logs
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:                        # skip blank lines
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.task.logs.append(f"[{timestamp}]  {line}")

    def flush(self):
        # Flush any remaining text that had no trailing newline
        if self._buffer.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.task.logs.append(f"[{timestamp}]  {self._buffer.strip()}")
            self._buffer = ""
        if self.original_stdout:
            self.original_stdout.flush()

# ==================== DETECTION WORKER ====================
def run_detection_task(task: PredictionTask, stop_event: threading.Event):
    """Run the full detection pipeline for a task"""
    import sys

    log_capture = LogCapture(task)

    try:
        task.status     = "running"
        task.started_at = datetime.now().isoformat()

        # Redirect stdout / stderr
        log_capture.original_stdout = sys.stdout
        log_capture.original_stderr = sys.stderr
        sys.stdout = log_capture
        sys.stderr = log_capture

        # ── Step 1: Classify uploaded video ──────────────────────
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ── STEP 1/3: Video Classification ──")
        classifier_output = classify_video(task.video_path)
        task.classifier_output = classifier_output
        task.logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}]  ✔ Classifier done → {classifier_output['detected']}"
        )

        # ── Step 2: Agent model routing ───────────────────────────
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ── STEP 2/3: Model Routing ──")
        agent = AlertAgent()
        selected_models = agent.decide_models(classifier_output)
        task.selected_models = [key for key, _ in selected_models]
        task.logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}]  ✔ Models selected: {task.selected_models}"
        )

        # ── Step 3: Detection ─────────────────────────────────────
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]  ── STEP 3/3: Detection ──")

        event_queue = queue.Queue()
        agent_ready = threading.Event()

        agent_thread = threading.Thread(
            target=agent.run,
            args=(event_queue,),
            daemon=True
        )
        agent_thread.start()
        agent_ready.set()

        output_path             = OUTPUT_DIR / f"{task.task_id}_output.mp4"

        # Pass video_path and stop_event so predict.py uses the correct file
        run_detection(
            event_queue,
            selected_models,
            video_path=task.video_path,    # ← FIX: use uploaded file, not config VIDEO_PATH
            stop_event=stop_event          # ← FIX: stop button support
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
        # Restore stdout / stderr
        sys.stdout = log_capture.original_stdout
        sys.stderr = log_capture.original_stderr

        # Clean up temp video file
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

# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Save the uploaded video to a system temp file (no uploads/ folder).
    Returns the temp file path so /predict can use it directly.
    """
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(400, "Only video files are allowed (.mp4 .avi .mov .mkv)")

    suffix = Path(file.filename).suffix
    # NamedTemporaryFile with delete=False → we clean it up after detection
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
    finally:
        tmp.close()

    return {"message": "Video uploaded", "path": tmp.name, "original_name": file.filename}

@app.post("/predict")
async def start_prediction(
    background_tasks: BackgroundTasks,
    request: PredictRequest
):
    """Start detection on the uploaded video (temp file path)"""
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
    """Stop a running detection task"""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    task = tasks[task_id]
    if task.status not in ("running", "pending"):
        return {"message": f"Task is already {task.status}", "task_id": task_id}

    stop_event = stop_flags.get(task_id)
    if stop_event:
        stop_event.set()
        task.logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}]  ⏹ Stop requested by user"
        )

    return {"message": "Stop signal sent", "task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    task = tasks[task_id]
    return TaskStatus(
        task_id           = task.task_id,
        status            = task.status,
        classifier_output = task.classifier_output,
        selected_models   = task.selected_models,
        results           = task.results,
        output_video      = task.output_video,
        error             = task.error
    )

@app.get("/logs/{task_id}")
async def stream_logs(task_id: str):
    """Stream logs in real-time using Server-Sent Events"""
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

    return FileResponse(
        task.output_video,
        media_type="video/mp4",
        filename=f"detection_{task_id}.mp4"
    )

@app.get("/analytics")
async def get_analytics():
    history      = load_history()
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
        "total_detections":      len(history),
        "status_counts":         status_counts,
        "model_usage":           model_usage,
        "recent_detections":     history[-10:],
        "detections_over_time":  detections_over_time[-20:]
    }

@app.delete("/history")
async def clear_history():
    """Delete all detection history"""
    if DB_FILE.exists():
        DB_FILE.unlink()
    return {"message": "History cleared"}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import logging

    class FilterAnalytics(logging.Filter):
        def filter(self, record):
            return 'GET /analytics' not in record.getMessage()

    logging.getLogger("uvicorn.access").addFilter(FilterAnalytics())

    print("=" * 70)
    print("AI DETECTION DASHBOARD SERVER")
    print("=" * 70)
    print("Starting server at http://localhost:8000")
    print("Open your browser and navigate to the URL above")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)