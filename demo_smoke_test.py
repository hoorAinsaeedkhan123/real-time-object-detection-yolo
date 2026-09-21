#!/usr/bin/env python3
"""
Smoke test: Load YOLOv8 model, verify it works on dummy frame.
"""
import sys
import numpy as np
from detector import ObjectDetector

print("🔬 Starting smoke test...")

# Test 1: Load model
print("\n1️⃣  Loading YOLOv8n model...")
detector = ObjectDetector(model_name="yolov8n")
if not detector.load_model():
    print("❌ Model load failed")
    sys.exit(1)

print("✅ Model loaded successfully")

# Test 2: Inference on dummy frame
print("\n2️⃣  Running inference on dummy frame (640x480)...")
dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

try:
    detections = detector.predict_frame(dummy_frame, conf=0.5)
    print(f"✅ Inference complete: {len(detections)} detections found")
except Exception as e:
    print(f"❌ Inference failed: {e}")
    sys.exit(1)

# Test 3: Model info
print("\n3️⃣  Model info:")
info = detector.get_model_info()
for k, v in info.items():
    print(f"   {k}: {v}")

print("\n✅ Smoke test PASSED - model ready for use")
