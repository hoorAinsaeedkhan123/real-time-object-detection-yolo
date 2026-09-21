# detector.py - YOLOv8 detection wrapper
import os
from ultralytics import YOLO
import numpy as np
from config import MODEL_NAME, TARGET_CLASSES, MODELS_DIR
from utils import format_timestamp, validate_coordinates

class ObjectDetector:
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize YOLO detector with specified model."""
        self.model_name = model_name
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
        self.model = None
        self.device = "cpu"  # Use CPU for portability
        
    def load_model(self):
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"✅ Model loaded: {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def predict_frame(self, frame: np.ndarray, conf: float = 0.5) -> list:
        """
        Run inference on single frame.
        Returns list of detection records.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            results = self.model(frame, conf=conf, verbose=False)
            detections = []
            
            for r in results:
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls.item())
                    if cls_id in TARGET_CLASSES:
                        conf_score = box.conf.item()
                        coords = box.xyxy.cpu().numpy()[0]  # x1, y1, x2, y2
                        
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        if validate_coordinates(x1, y1, x2, y2, frame.shape[1], frame.shape[0]):
                            detections.append({
                                'class_id': cls_id,
                                'class_name': TARGET_CLASSES[cls_id],
                                'confidence': float(conf_score),
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                            })
            
            return detections
        except Exception as e:
            print(f"❌ Inference error: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """Return model metadata."""
        if self.model is None:
            return {'status': 'not_loaded', 'model_name': self.model_name}
        return {
            'status': 'loaded',
            'model_name': self.model_name,
            'device': str(self.device),
            'target_classes': TARGET_CLASSES
        }
