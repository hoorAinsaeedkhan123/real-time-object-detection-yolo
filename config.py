# config.py - Configuration constants
import os

# Model config
MODEL_NAME = "yolov8n"
CONFIDENCE_DEFAULT = 0.45
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# Target classes from COCO
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# File handling
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
DATA_DIR = "data"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Video constraints
MAX_VIDEO_SIZE_MB = 500
MIN_FRAME_SIZE = (160, 120)
SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.flv', '.mkv'}

# Detection record fields
DETECTION_FIELDS = ['frame_index', 'timestamp_sec', 'timestamp_text', 
                    'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']

# Evaluation
IOU_THRESHOLD = 0.50
