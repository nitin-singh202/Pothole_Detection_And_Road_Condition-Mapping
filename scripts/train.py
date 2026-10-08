"""
YOLOv8 Model Training and Fine-Tuning Pipeline.

Configures and trains an Ultralytics YOLOv8 detector using transfer learning
from pretrained COCO checkpoints. Automatically selects CUDA GPU if available,
with graceful fallback to CPU. Saves the best checkpoint to models/ directory.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.utils.logger import setup_logger
from config import Config

logger = setup_logger("train_yolov8")


def get_optimal_device(requested_device: str = None) -> str:
    """Detects available hardware and resolves optimal execution device."""
    if requested_device and requested_device.lower() != "auto":
        return requested_device

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA GPU detected: {device_name}. Accelerating training on GPU:0")
        return "0"
    else:
        logger.info("CUDA GPU unavailable. Defaulting to CPU execution.")
        return "cpu"


def train_pothole_detector(
    data_yaml: Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    project_dir: Path = None,
    run_name: str = "pothole_run",
    export_best_to: Path = None,
) -> Path:
    """
    Executes YOLOv8 fine-tuning workflow.

    Args:
        data_yaml: Path to dataset configuration YAML.
        base_model: Name or path of pretrained baseline weights (e.g. 'yolov8n.pt').
        epochs: Number of training iterations over the dataset.
        imgsz: Square input resolution for inference/training.
        batch_size: Number of images processed per gradient update.
        device: Hardware device ('0', 'cpu', or 'auto').
        project_dir: Root directory for Ultralytics experiment artifacts.
        run_name: Experiment subdirectory name.
        export_best_to: Destination Path to copy the best.pt weights file.

    Returns:
        Path: Path to the exported best model weights.
    """
    resolved_device = get_optimal_device(device)
    target_project = project_dir or (BASE_DIR / "runs" / "train")
    target_export = export_best_to or (BASE_DIR / "models" / "pothole_yolov8n.pt")

    logger.info("=" * 70)
    logger.info("Starting YOLOv8 Fine-Tuning Pipeline")
    logger.info(f"  • Base Checkpoint : {base_model}")
    logger.info(f"  • Dataset Config  : {data_yaml}")
    logger.info(f"  • Epochs          : {epochs}")
    logger.info(f"  • Image Size      : {imgsz}x{imgsz}")
    logger.info(f"  • Batch Size      : {batch_size}")
    logger.info(f"  • Execution Device: {resolved_device}")
    logger.info("=" * 70)

    # 1. Initialize YOLO Model with Pretrained Weights
    logger.info(f"Loading pretrained weights [{base_model}]...")
    model = YOLO(base_model)

    start_time = time.time()

    # 2. Execute Training
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=resolved_device,
        project=str(target_project),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        plots=True,
    )

    elapsed_time_sec = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time_sec / 60.0:.2f} minutes.")

    # 3. Locate Best Weights
    train_dir = target_project / run_name
    best_weights_path = train_dir / "weights" / "best.pt"

    if best_weights_path.exists():
        target_export.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights_path, target_export)
        logger.info(f"Best model weights successfully saved to: {target_export}")
    else:
        logger.warning(f"best.pt not found at expected path: {best_weights_path}. Using last.pt fallback if available.")
        last_weights_path = train_dir / "weights" / "last.pt"
        if last_weights_path.exists():
            shutil.copy2(last_weights_path, target_export)

    # 4. Record Training Metadata Audit
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": base_model,
        "data_yaml": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": resolved_device,
        "training_duration_seconds": round(elapsed_time_sec, 2),
        "exported_weights": str(target_export),
    }

    meta_file = target_export.parent / "training_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Training metadata recorded to: {meta_file}")

    return target_export


def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune YOLOv8 Pothole Detector.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Baseline model: yolov8n.pt / yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, 0")
    parser.add_argument("--export", type=str, default="models/pothole_yolov8n.pt", help="Export destination")
    args = parser.parse_args()

    yaml_path = (BASE_DIR / args.data).resolve()
    export_path = (BASE_DIR / args.export).resolve()

    train_pothole_detector(
        data_yaml=yaml_path,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        export_best_to=export_path,
    )


if __name__ == "__main__":
    main()
