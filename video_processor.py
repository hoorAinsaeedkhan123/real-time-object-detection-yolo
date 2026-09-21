# video_processor.py - Video I/O and frame processing
import cv2
import os
from pathlib import Path
from config import SUPPORTED_FORMATS, MIN_FRAME_SIZE
from utils import format_timestamp

class VideoValidator:
    @staticmethod
    def validate_file(filepath: str) -> tuple:
        """
        Validate video file format and readability.
        Returns: (is_valid, error_message, metadata)
        """
        if not os.path.exists(filepath):
            return False, "File not found", {}
        
        ext = Path(filepath).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return False, f"Unsupported format. Supported: {SUPPORTED_FORMATS}", {}
        
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return False, "Cannot open video file with OpenCV", {}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            if frame_count == 0 or fps == 0:
                return False, "Invalid video: 0 frames or 0 FPS", {}
            if width < MIN_FRAME_SIZE[0] or height < MIN_FRAME_SIZE[1]:
                return False, f"Frame size too small: {width}x{height}", {}
            
            metadata = {
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration_sec': frame_count / fps if fps > 0 else 0
            }
            return True, "", metadata
        except Exception as e:
            return False, f"Validation error: {str(e)}", {}

class VideoReader:
    def __init__(self, filepath: str):
        """Initialize video reader."""
        self.filepath = filepath
        self.cap = None
        self.metadata = {}
        
    def open(self) -> bool:
        """Open video file."""
        try:
            self.cap = cv2.VideoCapture(self.filepath)
            if not self.cap.isOpened():
                return False
            
            self.metadata = {
                'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC))
            }
            return True
        except Exception as e:
            print(f"❌ Error opening video: {e}")
            return False
    
    def read_frame(self) -> tuple:
        """Read next frame. Returns (success, frame, frame_index)."""
        if self.cap is None or not self.cap.isOpened():
            return False, None, -1
        
        ret, frame = self.cap.read()
        if ret:
            frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            return True, frame, frame_idx
        return False, None, -1
    
    def get_metadata(self) -> dict:
        """Get video metadata."""
        return self.metadata.copy()
    
    def close(self):
        """Release video file."""
        if self.cap is not None:
            self.cap.release()
    
    def __del__(self):
        self.close()

class VideoWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float):
        """Initialize video writer."""
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        
    def open(self) -> bool:
        """Open video writer."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
            return self.writer.isOpened()
        except Exception as e:
            print(f"❌ Error opening writer: {e}")
            return False
    
    def write_frame(self, frame) -> bool:
        """Write frame to output video."""
        if self.writer is None or not self.writer.isOpened():
            return False
        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            print(f"❌ Error writing frame: {e}")
            return False
    
    def close(self):
        """Release writer."""
        if self.writer is not None:
            self.writer.release()
    
    def __del__(self):
        self.close()

def draw_detections(frame, detections: list, draw_conf: bool = True) -> None:
    """
    Draw bounding boxes and labels on frame (in-place).
    """
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        label = det['class_name']
        conf = det['confidence']
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        text = f"{label} {conf:.2f}" if draw_conf else label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (x1, y1 - 5), font, 0.6, (0, 255, 0), 2)
