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
        "polygon_file": "animal_boundary/deer1_boundary.json",
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
        "polygon_file": "animal_boundary/leopard_boundary.json",
    },
    "tiger": {
        "model_path": "weights/tiger_best.pt",
        "target_classes": ["tiger"],
        "polygon_file": "animal_boundary/tiger_boundary.json",
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
