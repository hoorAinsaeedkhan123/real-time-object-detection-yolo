# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                   Streamlit Web UI (app.py)                      │
│  - File upload / Configuration sidebar / Results display         │
└──────────────────┬──────────────────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │ Video Validation  │
         │ (VideoValidator)  │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────────────────┐
         │  Video I/O Pipeline           │
         │ ┌──────────────────────────┐  │
         │ │ VideoReader (OpenCV)     │  │
         │ │ - Frame iteration        │  │
         │ │ - Metadata extraction    │  │
         │ └──────────────────────────┘  │
         │                                │
         │ ┌──────────────────────────┐  │
         │ │ VideoWriter (OpenCV)     │  │
         │ │ - MP4 encoding           │  │
         │ │ - Frame composition       │  │
         │ └──────────────────────────┘  │
         └─────────┬─────────────────────┘
                   │
         ┌─────────▼──────────────────┐
         │  Object Detector (YOLO)    │
         │  (ObjectDetector)          │
         │  - Model: YOLOv8n.pt       │
         │  - Device: CPU/CUDA        │
         │  - Inference: 640x640      │
         └─────────┬──────────────────┘
                   │
         ┌─────────▼────────────────┐
         │  Annotation & Drawing    │
         │ - Bounding boxes         │
         │ - Class labels           │
         │ - Confidence scores      │
         │ - Timestamps             │
         └─────────┬────────────────┘
                   │
         ┌─────────▼──────────────────┐
         │ Evaluation & Metrics       │
         │ (EvaluationMetrics)        │
         │ - IoU-based matching       │
         │ - TP/FP/FN calculation     │
         │ - Precision/Recall/F1      │
         └────────────────────────────┘
```

## Module Breakdown

### 1. app.py - Web Interface

**Role:** User-facing Streamlit application

**Responsibilities:**
- File upload & validation
- Configuration UI (confidence, classes)
- Real-time progress tracking
- Results summary & download

**Key Functions:**
- `st.file_uploader()` - Handle user video upload
- `st.slider()` - Confidence threshold control
- `st.download_button()` - Output video download

**Flow:**
```
Upload → Validate → Configure → Process → Display Results
```

### 2. detector.py - YOLOv8 Wrapper

**Role:** Unified interface to model inference

**Class: ObjectDetector**

```python
detector = ObjectDetector(model_name="yolov8n")
detector.load_model()
detections = detector.predict_frame(frame, conf=0.45)
```

**Output Format:**
```python
[
  {
    'class_id': 0,
    'class_name': 'person',
    'confidence': 0.87,
    'x1': 100, 'y1': 120, 'x2': 200, 'y2': 300
  },
  ...
]
```

**Key Features:**
- Lazy loading (model downloaded on first use)
- CPU-safe inference (no CUDA dependency)
- Target class filtering
- Structured detection records

### 3. video_processor.py - OpenCV Integration

**Three Main Classes:**

#### VideoValidator
Checks file validity before processing:
```python
is_valid, error_msg, metadata = VideoValidator.validate_file("video.mp4")
# metadata = {frame_count, fps, width, height, duration_sec}
```

#### VideoReader
Frame-by-frame iteration:
```python
reader = VideoReader("input.mp4")
reader.open()
while True:
    ret, frame, frame_idx = reader.read_frame()
    if not ret:
        break
reader.close()
```

#### VideoWriter
Annotated output creation:
```python
writer = VideoWriter("output.mp4", width, height, fps)
writer.open()
writer.write_frame(annotated_frame)
writer.close()
```

#### draw_detections()
In-place frame annotation:
```python
draw_detections(frame, detections, draw_conf=True)
# Modifies frame with boxes, labels, confidence
```

### 4. evaluation.py - Metrics & Evaluation

**Class: EvaluationMetrics**

Implements standard COCO evaluation:
```python
metrics = EvaluationMetrics(iou_threshold=0.50)
tp, fp, fn = metrics.match_boxes(ground_truth, predictions)
results = metrics.calculate_metrics(tp, fp, fn)
# results = {precision, recall, f1, ...}
```

**Algorithm:**
1. For each prediction, find best-matching GT box (same class, highest IoU)
2. If IoU ≥ threshold, count as TP; else FP
3. Unmatched GT boxes are FN
4. Calculate metrics from aggregates

**Class: EvaluationReporter**

Persistent evaluation logging:
```python
reporter = EvaluationReporter("evaluation/results.csv")
reporter.add_result(image_id, ..., tp, fp, fn, precision, recall, f1, ...)
reporter.save_csv()
summary = reporter.summary_stats()
```

### 5. config.py - Constants & Settings

Centralized configuration:

```python
MODEL_NAME = "yolov8n"
CONFIDENCE_DEFAULT = 0.45
TARGET_CLASSES = {0: "person", 1: "bicycle", ...}
IOU_THRESHOLD = 0.50
```

**Design rationale:** Single source of truth; easy to tweak without editing code.

### 6. utils.py - Helper Functions

| Function | Purpose |
|----------|---------|
| `sanitize_filename()` | Safe path generation |
| `generate_safe_path()` | Hash-based collision avoidance |
| `format_timestamp()` | Frame index → HH:MM:SS |
| `validate_coordinates()` | Boundary checking |
| `calculate_iou()` | Box overlap metric |

## Data Flow Diagram

### Processing Pipeline

```
User uploads video.mp4
    ↓
VideoValidator.validate_file()
    ├─ Check format (.mp4, .avi, etc.)
    ├─ Verify OpenCV can read
    └─ Extract metadata (FPS, size, frame_count)
    ↓
VideoReader.open()
    ↓
For each frame:
    ├─ VideoReader.read_frame()
    ├─ ObjectDetector.predict_frame(frame, conf=0.45)
    ├─ Filter by target classes
    ├─ Add timestamp
    ├─ draw_detections() in-place
    ├─ VideoWriter.write_frame()
    └─ Accumulate detection records
    ↓
VideoWriter.close()
    ↓
Output video + detection log
    ↓
Streamlit displays:
    ├─ Metrics (frames, detections, speed)
    ├─ Detection table
    └─ Download button
```

### Inference Loop Pseudocode

```python
detector.load_model()
reader.open()
writer.open()

detections_all = []
for frame_idx in range(frame_count):
    frame = reader.read_frame()
    
    # Inference
    dets = detector.predict_frame(frame, conf=confidence)
    dets = filter_by_classes(dets, target_classes)
    
    # Timestamp
    ts_sec, ts_text = format_timestamp(frame_idx, fps)
    for det in dets:
        det['timestamp_text'] = ts_text
    
    # Annotate & write
    draw_detections(frame, dets)
    add_timestamp_label(frame, ts_text)
    writer.write_frame(frame)
    
    detections_all.extend(dets)

reader.close()
writer.close()
return detections_all, output_path
```

## Error Handling Strategy

### Graceful Degradation

```python
try:
    detector.load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    return

try:
    ret, frame, _ = reader.read_frame()
    if not ret:
        break  # End of video
except Exception as e:
    st.error(f"Frame read error: {e}")
    break
```

### User-Friendly Messages

| Error Type | Message | Action |
|------------|---------|--------|
| File not found | "File not found" | Show upload form |
| Invalid format | "Unsupported format. Supported: .mp4, .avi..." | Suggest ffmpeg conversion |
| Model download fails | "Model download failed. Check internet." | Retry button |
| Corrupt frame | "Skipping corrupt frame X" | Continue with next frame |

## Performance Characteristics

### Memory Usage

- **Model:** ~330 MB (YOLOv8n in memory)
- **Single frame:** ~1 MB (640x480 RGB)
- **Total:** ~500 MB for typical usage

### Processing Speed

| Stage | Time per frame (CPU i7) |
|-------|------------------------|
| Read frame | ~5 ms |
| Inference | ~80–120 ms |
| Draw annotations | ~10 ms |
| Write frame | ~20 ms |
| **Total** | **~120–160 ms** |
| **FPS** | **6–8 FPS** |

## Extensibility Points

### Easy to Add

1. **Different model:**
   ```python
   MODEL_NAME = "yolov8l"  # Larger model
   ```

2. **Custom class filtering:**
   ```python
   dets = [d for d in dets if d['class_name'] in my_classes]
   ```

3. **Bounding box filtering:**
   ```python
   dets = [d for d in dets if (d['x2'] - d['x1']) > min_width]
   ```

4. **Post-processing:**
   ```python
   dets = apply_nms(dets, iou_threshold=0.3)
   ```

### Harder to Add (Requires Refactor)

- GPU support (possible but not yet implemented)
- Temporal tracking across frames (requires state management)
- Custom model training (out of scope for prototype)
- Real-time streaming input (would need different I/O)

---

**Architecture Version:** 1.0  
**Last Updated:** 2026-09-08  
**Status:** Stable
