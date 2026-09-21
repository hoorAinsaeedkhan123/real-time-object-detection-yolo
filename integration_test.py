#!/usr/bin/env python3
"""
Integration test: Create a synthetic test video, process it, verify output.
"""
import os
import cv2
import numpy as np
import sys
from pathlib import Path
from detector import ObjectDetector
from video_processor import VideoValidator, VideoReader, VideoWriter, draw_detections
from utils import format_timestamp

def create_synthetic_video(output_path, num_frames=30, width=640, height=480, fps=30):
    """Generate a simple synthetic video for testing."""
    print(f"📝 Creating synthetic video: {output_path}")
    
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    if not writer.isOpened():
        print("❌ Failed to open video writer")
        return False
    
    for frame_idx in range(num_frames):
        # Create frame with changing content
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Add some moving rectangles (synthetic objects)
        offset = (frame_idx * 5) % width
        cv2.rectangle(frame, (offset, 100), (offset + 80, 200), (0, 255, 0), -1)
        cv2.rectangle(frame, ((offset + 200) % width, 300), 
                     ((offset + 280) % width, 400), (255, 0, 0), -1)
        
        # Add text
        ts_sec, ts_text = format_timestamp(frame_idx, fps)
        cv2.putText(frame, f"Frame {frame_idx}: {ts_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
    
    writer.release()
    print(f"✅ Synthetic video created: {output_path}")
    return True

def test_end_to_end():
    """Full end-to-end test."""
    print("🧪 Integration Test: End-to-End Video Processing\n")
    
    # Setup
    test_dir = "test_artifacts"
    os.makedirs(test_dir, exist_ok=True)
    
    input_video = os.path.join(test_dir, "test_input.mp4")
    output_video = os.path.join(test_dir, "test_output.mp4")
    
    # Step 1: Create synthetic video
    print("=" * 60)
    print("STEP 1: Create Synthetic Test Video")
    print("=" * 60)
    if not create_synthetic_video(input_video, num_frames=20, width=640, height=480, fps=30):
        return False
    
    # Step 2: Validate
    print("\n" + "=" * 60)
    print("STEP 2: Validate Input Video")
    print("=" * 60)
    is_valid, error_msg, metadata = VideoValidator.validate_file(input_video)
    if not is_valid:
        print(f"❌ Validation failed: {error_msg}")
        return False
    
    print(f"✅ Video valid:")
    print(f"   Size: {metadata['width']}x{metadata['height']}")
    print(f"   FPS: {metadata['fps']}")
    print(f"   Frames: {metadata['frame_count']}")
    print(f"   Duration: {metadata['duration_sec']:.2f}s")
    
    # Step 3: Load detector
    print("\n" + "=" * 60)
    print("STEP 3: Load YOLOv8 Detector")
    print("=" * 60)
    detector = ObjectDetector(model_name="yolov8n")
    if not detector.load_model():
        print("❌ Model load failed")
        return False
    print("✅ Model loaded")
    
    # Step 4: Process video
    print("\n" + "=" * 60)
    print("STEP 4: Process Video (Frame-by-Frame)")
    print("=" * 60)
    
    reader = VideoReader(input_video)
    if not reader.open():
        print("❌ Failed to open video for reading")
        return False
    
    writer = VideoWriter(output_video, metadata['width'], metadata['height'], metadata['fps'])
    if not writer.open():
        print("❌ Failed to open writer")
        return False
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame, frame_idx = reader.read_frame()
        if not ret:
            break
        
        # Run inference
        detections = detector.predict_frame(frame, conf=0.45)
        
        # Add timestamp
        ts_sec, ts_text = format_timestamp(frame_idx, metadata['fps'])
        for det in detections:
            det['timestamp_text'] = ts_text
        
        # Draw detections
        draw_detections(frame, detections, draw_conf=True)
        cv2.putText(frame, f"[{ts_text}]", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Write frame
        writer.write_frame(frame)
        
        frame_count += 1
        detection_count += len(detections)
        print(f"  Frame {frame_count}/{metadata['frame_count']}: "
             f"{len(detections)} detections")
    
    reader.close()
    writer.close()
    
    print(f"✅ Processing complete: {frame_count} frames, {detection_count} detections")
    
    # Step 5: Verify output
    print("\n" + "=" * 60)
    print("STEP 5: Verify Output Video")
    print("=" * 60)
    
    if not os.path.exists(output_video):
        print("❌ Output video not created")
        return False
    
    output_size = os.path.getsize(output_video) / (1024 * 1024)
    print(f"✅ Output video created: {output_size:.2f} MB")
    
    # Can read output
    is_valid_out, error_msg_out, metadata_out = VideoValidator.validate_file(output_video)
    if not is_valid_out:
        print(f"❌ Output validation failed: {error_msg_out}")
        return False
    
    print(f"✅ Output valid:")
    print(f"   Frames: {metadata_out['frame_count']}")
    print(f"   FPS: {metadata_out['fps']}")
    
    # Sanity checks
    if metadata_out['frame_count'] != metadata['frame_count']:
        print(f"⚠️  Frame count mismatch: {metadata_out['frame_count']} vs {metadata['frame_count']}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 60)
    print(f"\nArtifacts saved to: {test_dir}/")
    return True

if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
