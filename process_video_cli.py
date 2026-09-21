#!/usr/bin/env python3
"""
CLI video processor: process video end-to-end from command line.
Usage: python process_video_cli.py --input <video> --output <output.mp4> [--conf 0.45] [--classes 0,2,7]
"""
import argparse
import sys
import time
from pathlib import Path
from detector import ObjectDetector
from video_processor import VideoValidator, VideoReader, VideoWriter, draw_detections
from utils import format_timestamp
from config import TARGET_CLASSES

def main():
    parser = argparse.ArgumentParser(description="Process video with YOLOv8 detection")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--classes", type=str, default="0,1,2,3,5,7",
                       help="Comma-separated class IDs (default: person,bicycle,car,motorcycle,bus,truck)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Validate input
    print(f"📹 Input: {args.input}")
    print(f"📊 Confidence: {args.conf}")
    print(f"🎯 Target classes: {args.classes}")
    
    is_valid, error_msg, metadata = VideoValidator.validate_file(args.input)
    if not is_valid:
        print(f"❌ Validation failed: {error_msg}")
        return 1
    
    print(f"✅ Valid video: {metadata['width']}x{metadata['height']} @ {metadata['fps']:.1f}FPS")
    print(f"   Frames: {metadata['frame_count']}, Duration: {metadata['duration_sec']:.1f}s")
    
    # Parse target classes
    try:
        target_class_ids = list(map(int, args.classes.split(",")))
    except ValueError:
        print(f"❌ Invalid class IDs: {args.classes}")
        return 1
    
    # Load detector
    print("\n🤖 Loading detector...")
    detector = ObjectDetector(model_name="yolov8n")
    if not detector.load_model():
        print("❌ Failed to load model")
        return 1
    
    # Process
    print(f"\n▶️  Processing...")
    reader = VideoReader(args.input)
    if not reader.open():
        print("❌ Failed to open video")
        return 1
    
    writer = VideoWriter(args.output, metadata['width'], metadata['height'], metadata['fps'])
    if not writer.open():
        print("❌ Failed to open writer")
        return 1
    
    all_detections = []
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame, frame_idx = reader.read_frame()
            if not ret:
                break
            
            # Detect
            detections = detector.predict_frame(frame, conf=args.conf)
            detections = [d for d in detections if d['class_id'] in target_class_ids]
            
            # Timestamp
            ts_sec, ts_text = format_timestamp(frame_idx, metadata['fps'])
            for det in detections:
                det['timestamp_sec'] = ts_sec
                det['timestamp_text'] = ts_text
                det['frame_index'] = frame_idx
            all_detections.extend(detections)
            
            # Draw
            draw_detections(frame, detections, draw_conf=True)
            cv2.putText(frame, f"[{ts_text}]", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Write
            writer.write_frame(frame)
            
            frame_count += 1
            detection_count += len(detections)
            
            if frame_count % max(1, metadata['frame_count'] // 10) == 0:
                progress = 100 * frame_count / metadata['frame_count']
                print(f"  {progress:5.1f}% | Frame {frame_count}/{metadata['frame_count']} | "
                     f"Detections: {detection_count}")
    
    finally:
        reader.close()
        writer.close()
    
    elapsed = time.time() - start_time
    fps_processed = frame_count / elapsed if elapsed > 0 else 0
    
    # Summary
    print(f"\n✅ Processing complete!")
    print(f"   Frames: {frame_count}")
    print(f"   Detections: {detection_count}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Speed: {fps_processed:.1f} FPS")
    print(f"   Output: {args.output}")
    
    return 0

if __name__ == "__main__":
    import cv2
    sys.exit(main())
