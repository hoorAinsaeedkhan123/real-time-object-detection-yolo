# app.py - Streamlit UI for object detection
import streamlit as st
import cv2
import os
import time
from pathlib import Path
from detector import ObjectDetector
from video_processor import VideoValidator, VideoReader, VideoWriter, draw_detections
from utils import generate_safe_path, format_timestamp
from config import TARGET_CLASSES, CONFIDENCE_DEFAULT, UPLOAD_DIR, OUTPUT_DIR

st.set_page_config(page_title="Object Detection for CCTV", layout="wide")
st.title("🎥 Object Detection for Security Footage")
st.markdown("Upload a video, configure detection parameters, and analyze for objects.")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Configuration")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, CONFIDENCE_DEFAULT, 0.05)
    
    selected_classes = st.multiselect(
        "Target Classes",
        options=list(TARGET_CLASSES.values()),
        default=list(TARGET_CLASSES.values())[:3]
    )
    target_class_ids = [k for k, v in TARGET_CLASSES.items() if v in selected_classes]
    
    st.markdown("---")
    st.markdown("**Model Info**")
    st.write(f"Model: YOLOv8n (nano)")
    st.write(f"Input size: 640x640")
    st.write(f"Device: CPU")

# Main workflow
tab1, tab2 = st.tabs(["Process Video", "About"])

with tab1:
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv", "flv"])
    
    if uploaded_file is not None:
        # Save uploaded file
        safe_path = generate_safe_path(UPLOAD_DIR, uploaded_file.name)
        with open(safe_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File saved: {Path(safe_path).name}")
        
        # Validate
        is_valid, error_msg, metadata = VideoValidator.validate_file(safe_path)
        if not is_valid:
            st.error(f"❌ Validation failed: {error_msg}")
        else:
            st.info(f"Video: {metadata['width']}x{metadata['height']} @ {metadata['fps']:.1f}FPS, "
                   f"{metadata['frame_count']} frames ({metadata['duration_sec']:.1f}s)")
            
            # Process button
            if st.button("🚀 Process Video", key="process_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                detector = ObjectDetector()
                if not detector.load_model():
                    st.error("❌ Failed to load model")
                else:
                    reader = VideoReader(safe_path)
                    if not reader.open():
                        st.error("❌ Failed to open video for reading")
                    else:
                        meta = reader.get_metadata()
                        output_path = os.path.join(OUTPUT_DIR, 
                                                  f"{Path(safe_path).stem}_detected.mp4")
                        writer = VideoWriter(output_path, meta['width'], meta['height'], meta['fps'])
                        
                        if not writer.open():
                            st.error("❌ Failed to initialize writer")
                        else:
                            all_detections = []
                            frame_count = 0
                            detection_count = 0
                            start_time = time.time()
                            
                            while True:
                                ret, frame, frame_idx = reader.read_frame()
                                if not ret:
                                    break
                                
                                # Run detection
                                detections = detector.predict_frame(frame, conf=conf_threshold)
                                detections = [d for d in detections if d['class_id'] in target_class_ids]
                                
                                # Add timestamp
                                ts_sec, ts_text = format_timestamp(frame_idx, meta['fps'])
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
                                progress = min(frame_count / meta['frame_count'], 1.0)
                                progress_bar.progress(progress)
                                status_text.text(f"Frame {frame_count}/{meta['frame_count']} | "
                                               f"Detections: {detection_count}")
                            
                            reader.close()
                            writer.close()
                            
                            elapsed = time.time() - start_time
                            fps_processed = frame_count / elapsed if elapsed > 0 else 0
                            
                            st.success("✅ Processing complete!")
                            st.balloons()
                            
                            # Results summary
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Frames Processed", frame_count)
                            with col2:
                                st.metric("Total Detections", detection_count)
                            with col3:
                                st.metric("Processing Time", f"{elapsed:.1f}s")
                            with col4:
                                st.metric("Speed", f"{fps_processed:.1f} FPS")
                            
                            # Download link
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Annotated Video",
                                    data=file.read(),
                                    file_name=Path(output_path).name,
                                    mime="video/mp4"
                                )
                            
                            # Detections table
                            if all_detections:
                                st.subheader("📊 Detection Details")
                                import pandas as pd
                                df = pd.DataFrame(all_detections)
                                st.dataframe(df[['frame_index', 'timestamp_text', 'class_name', 
                                               'confidence']], use_container_width=True)

with tab2:
    st.markdown("""
    ### About This Project
    
    **Objective:** Detect objects (persons, vehicles) in security footage using YOLOv8.
    
    **Key Features:**
    - Frame-by-frame object detection with YOLOv8n
    - Configurable confidence threshold
    - Selectable target classes
    - Timestamped bounding boxes
    - Annotated output video download
    
    **Architecture:**
    - `detector.py`: YOLOv8 inference wrapper
    - `video_processor.py`: OpenCV video I/O
    - `evaluation.py`: Metrics and reporting
    - `config.py`: Constants and settings
    - `utils.py`: Helpers (timestamp, IoU, sanitization)
    
    **Model:** YOLOv8n (nano) - 3.2M parameters, COCO pretrained
    
    **Limitations:**
    - CPU inference is slower than GPU
    - Distant or occluded objects may be missed
    - No temporal tracking (frame-by-frame only)
    - Confidence threshold affects precision/recall tradeoff
    """)
