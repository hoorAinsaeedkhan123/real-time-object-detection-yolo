# utils.py - Utility functions
import os
import hashlib
import re
from pathlib import Path
from datetime import timedelta

def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize uploaded filename for safe storage."""
    basename = Path(filename).stem
    ext = Path(filename).suffix.lower()
    # Keep only alphanumeric, dash, underscore
    safe_name = re.sub(r'[^\w\-]', '', basename)[:max_length]
    return f"{safe_name}{ext}"

def generate_safe_path(upload_dir: str, filename: str) -> str:
    """Generate safe storage path with hash to prevent collisions."""
    safe_name = sanitize_filename(filename)
    hash_suffix = hashlib.sha256(filename.encode()).hexdigest()[:8]
    final_name = f"{safe_name[:-len(Path(filename).suffix)]}__{hash_suffix}{Path(filename).suffix}"
    return os.path.join(upload_dir, final_name)

def format_timestamp(frame_idx: int, fps: float) -> tuple:
    """
    Convert frame index to timestamp.
    Returns: (total_seconds, formatted_string)
    """
    if fps <= 0:
        return 0.0, "00:00:00"
    total_sec = frame_idx / fps
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    seconds = int(total_sec % 60)
    formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return total_sec, formatted

def validate_coordinates(x1, y1, x2, y2, img_width: int, img_height: int) -> bool:
    """Validate bounding box coordinates are within image."""
    if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
        return False
    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True

def calculate_iou(box1: tuple, box2: tuple) -> float:
    """
    Calculate Intersection over Union (IoU) for two boxes.
    box format: (x1, y1, x2, y2)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
