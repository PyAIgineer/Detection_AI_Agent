# ==================== CONFIG.PY ====================
# Single source of truth for all paths, model configs, and system parameters.
# Change model paths or video input ONLY here.

# ==================== INPUT ====================
VIDEO_PATH = "input_videos/fire_2.mp4"

# ==================== INDIVIDUAL MODEL WEIGHTS ====================
# Each key maps to a specific model file, target classes, and its own boundary polygon.
# Set polygon_file to None if no boundary crossing detection is needed for that model.
MODEL_CONFIGS = {
    "deer": {
        "model_path": "weights/deer_v2_best.pt",
        "target_classes": ["deer"],
        "polygon_file": "",
    },
    "elephant": {
        "model_path": "weights/elephant.pt",
        "target_classes": ["elephant"],
        "polygon_file": "animal_boundary/elephant1_boundary.json",
    },
    "fire_smoke": {
        "model_path": "weights/fire_l.pt",
        "target_classes": ["smoke", "fire"],
        "polygon_file": None,
    },
    "leopard": {
        "model_path": "weights/leopard_v1_best.pt",
        "target_classes": ["leopard"],
        "polygon_file": None,
    },
    "tiger": {
        "model_path": "weights/tiger_best.pt",
        "target_classes": ["tiger"],
        "polygon_file": "animal_boundary/tiger_boundary.json",
    },
    "person": {
        "model_path": "weights/yolo26s.pt",
        "target_classes": ["person"],
        "polygon_file": None,
    },
    "necklace": {
        "model_path": "weights/Necklace_v2.pt",
        "target_classes": ["necklace"],
        "polygon_file": None,
    },
    "earrings": {
        "model_path": "weights/earrings.pt",
        "target_classes": ["earrings"],
        "polygon_file": None,
    },
}

# ==================== DETECTION PARAMETERS ====================
CONF_THRESHOLD  = 0.4
IOU_THRESHOLD   = 0.45
RESIZE_RATIO    = 0.5
ALERT_COOLDOWN  = 60

# ==================== VIDEO OUTPUT ====================
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = "output_videos/detection_output.mp4"

# ==================== CLASSIFIER SETTINGS ====================
# Frame numbers to extract for pre-classification (1-indexed)
CLASSIFIER_FRAMES = [1, 15, 30]
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# ==================== RTSP CAMERAS ====================
# Set URL to "" to mark camera as unconfigured (will show placeholder tile).
# camera_id keys must stay cctv_01 … cctv_06 — the dashboard derives tile
# labels and stream endpoints directly from these keys.
RTSP_CAMERAS = {
    "cctv_01": "rtsp://admin:Honey@123@192.168.1.10:554/1/2",   # e.g. "rtsp://admin:pass@192.168.1.101:554/stream"
    # "cctv_02": "rtsp://admin:Honey@123@192.168.1.10:554/2/2",
    "cctv_02": "",
    "cctv_03": "",
    "cctv_04": "",
    "cctv_05": "",
    "cctv_06": "",
}
# When True, the dashboard auto-starts all configured RTSP streams on page load.
# Can be toggled at runtime via POST /rtsp/autostart/toggle without restarting the server.
RTSP_AUTO_START = True

# Seconds to wait before marking a dropped stream as FAILED (no auto-reconnect).
RTSP_CONNECT_TIMEOUT = 10