# Deployment Guide

## Local Development

### Quick Start

```bash
git clone <repo>
cd week3-object-detection-security
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Visit `http://localhost:8501`

## Streamlit Cloud Deployment

### Prerequisites

- GitHub account with repo pushed
- Streamlit Cloud account (free tier available)

### Steps

1. **Commit & push to GitHub:**
   ```bash
   git add .
   git commit -m "Week 3: Object detection prototype"
   git push origin main
   ```

2. **Create `streamlit/config.toml` (optional, for customization):**
   ```toml
   [client]
   toolbarMode = "minimal"
   
   [logger]
   level = "error"
   
   [theme]
   primaryColor = "#1f77b4"
   ```

3. **Deploy on Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select GitHub repo, branch, and script (`app.py`)
   - Click "Deploy"

4. **Monitor & logs:**
   - Check Streamlit Cloud dashboard
   - View app logs in real-time

### Environment Variables

Set on Streamlit Cloud settings:

```
MODEL_DEVICE=cpu
MAX_VIDEO_SIZE_MB=500
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run

```bash
docker build -t object-detection:latest .
docker run -p 8501:8501 -v $(pwd)/uploads:/app/uploads object-detection:latest
```

## GPU Deployment (Optional)

### CUDA Setup

1. Install NVIDIA CUDA Toolkit 12.1+
2. Replace `torch==2.14.0` with GPU variant in requirements.txt
3. Update `config.py`:
   ```python
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
   ```

### Performance Gains

- CPU: ~5–10 FPS (single video)
- GPU (RTX 4060): ~50–100 FPS

## Load Testing

### Synthetic Load

```bash
# Generate 10x test videos
for i in {1..10}; do
  python integration_test.py
done
```

### Benchmark

```bash
# Process 1-hour video on different hardware
time python process_video_cli.py --input long_video.mp4 --output out.mp4
```

Expected times:
- CPU (i7): ~3 hours
- GPU (RTX 4060): ~15 min

## Monitoring & Logging

### Streamlit Logs

```bash
# View logs
tail -f ~/.streamlit/logs/2026-09-08.log
```

### Custom Logging

Add to `config.py`:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Backup & Recovery

### Model Weights

Model (~167MB) auto-downloads to `models/yolov8n.pt`.

Backup:
```bash
cp models/yolov8n.pt models/yolov8n.pt.bak
```

### Output Preservation

Move outputs to archival:
```bash
tar -czf outputs_backup_$(date +%Y%m%d).tar.gz outputs/
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Streamlit takes forever to load" | Model downloading. Check `models/yolov8n.pt` exists. |
| 502 Bad Gateway on cloud | Memory limit hit. Upgrade instance or reduce batch size. |
| "No space left on device" | Clear `outputs/` and `uploads/`. Model cache: `rm ~/.cache/ultralytics`. |
| GPU out of memory | Reduce model size to YOLOv8n or use CPU. |

## Security Considerations

- Never commit `.env` with secrets
- Sanitize uploaded filenames (already done in `utils.py`)
- Validate video file content (not just extension)
- Run inference in sandboxed environment if handling untrusted video
- Set max upload size (already enforced in config)

## Performance Tuning

### Reduce Inference Time

1. **Lower confidence threshold:** Fewer boxes to draw
2. **Resize frames before inference:**
   ```python
   resized = cv2.resize(frame, (416, 416))  # YOLOv8n's native size
   ```
3. **Skip frames:** Process every Nth frame
4. **Use smaller model:** YOLOv8n → YOLOv5n (if compatibility allows)

### Optimize Output Video

```python
# Lower bitrate
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Use lower FPS if acceptable
```

## Maintenance

### Regular Tasks

- Monitor model accuracy on new footage
- Update Ultralytics if major releases available
- Review error logs monthly
- Archive old outputs

### Upgrade Path

1. Test new YOLOv8 version locally
2. Update `requirements.txt` & `config.py`
3. Run full test suite
4. Deploy incrementally

---

**Last updated:** 2026-09-08  
**Version:** 1.0  
**Status:** Production-ready
