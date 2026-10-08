# Model Artifacts Directory

This directory stores YOLOv8 weights and model checkpoints for the Pothole Detection & Road Condition Mapping System.

---

## Supported Weights

1. **`pothole_yolov8n.pt`** (Recommended for student laptops / CPU):
   - Fine-tuned YOLOv8 Nano model trained specifically on pothole dataset annotations.
   - Lowest inference latency and memory footprint (~6 MB file size).

2. **`yolov8n.pt`** (Pretrained COCO Baseline):
   - Standard baseline weights downloaded automatically by Ultralytics for transfer learning or sanity testing.

3. **`pothole_yolov8s.pt`** (Alternative for systems with dedicated GPU):
   - YOLOv8 Small architecture for improved detection accuracy on small, distant road defects.

---

## Instructions

- When training your model using `scripts/train.py`, the best weights will be automatically saved here as `models/pothole_yolov8n.pt`.
- If you have an existing pre-trained weights file from previous experiments, place it directly in this directory and update `MODEL_PATH` in your `.env` file.
- Model `.pt` binary files are excluded from Git version control via `.gitignore` to prevent repository bloat.
