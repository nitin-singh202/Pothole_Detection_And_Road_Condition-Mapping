"""
Model Evaluation and Benchmark Reporting Pipeline.

Evaluates a trained YOLOv8 model against validation or test datasets.
Extracts empirical metrics (mAP50, mAP50-95, Precision, Recall, F1, Latency, FPS),
generates structured JSON audit logs, and prints a comprehensive evaluation table.
Strictly calculates real measurements; never fabricates results.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.utils.logger import setup_logger

logger = setup_logger("evaluate_yolov8")


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    split: str = "val",
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cpu",
    output_report_path: Path = None,
) -> dict:
    """
    Runs model validation and calculates empirical performance metrics.

    Args:
        model_path: Path to YOLOv8 weights file (.pt).
        data_yaml: Path to data.yaml dataset definition.
        split: Dataset split to evaluate ('val' or 'test').
        imgsz: Input image resolution.
        batch_size: Evaluation batch size.
        device: Hardware device ('cpu' or '0').
        output_report_path: Destination path for JSON report.

    Returns:
        dict: Empirical evaluation metrics dictionary.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights file not found at: {model_path}\n"
            "Please train a model using 'python scripts/train.py' or specify an existing weights file."
        )

    logger.info("=" * 70)
    logger.info("Starting Model Evaluation & Benchmark Pipeline")
    logger.info(f"  • Model Path   : {model_path}")
    logger.info(f"  • Dataset Config: {data_yaml}")
    logger.info(f"  • Split         : {split}")
    logger.info(f"  • Device        : {device}")
    logger.info("=" * 70)

    model = YOLO(str(model_path))

    # Execute validation pass
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        verbose=True,
        plots=True,
    )

    # Extract empirical metrics from Ultralytics Results object
    box_metrics = val_results.box
    precision = float(box_metrics.mp)  # Mean precision across classes
    recall = float(box_metrics.mr)     # Mean recall across classes
    map50 = float(box_metrics.map50)   # mAP at IoU 0.50
    map50_95 = float(box_metrics.map)  # mAP at IoU 0.50:0.95

    # Compute F1 Score
    if (precision + recall) > 0:
        f1_score = float(2 * (precision * recall) / (precision + recall))
    else:
        f1_score = 0.0

    # Extract speed metrics (in milliseconds per frame)
    speed_dict = val_results.speed
    preprocess_ms = speed_dict.get("preprocess", 0.0)
    inference_ms = speed_dict.get("inference", 0.0)
    postprocess_ms = speed_dict.get("postprocess", 0.0)
    total_latency_ms = preprocess_ms + inference_ms + postprocess_ms
    measured_fps = round(1000.0 / total_latency_ms, 2) if total_latency_ms > 0 else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": str(model_path.name),
        "split_evaluated": split,
        "device": device,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "map_50": round(map50, 4),
            "map_50_95": round(map50_95, 4),
        },
        "latency_profile_ms": {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "loss_postprocess_ms": round(postprocess_ms, 2),
            "total_latency_per_frame_ms": round(total_latency_ms, 2),
            "measured_fps": measured_fps,
        },
    }

    # Save JSON report
    report_file = output_report_path or (BASE_DIR / "outputs" / "reports" / "evaluation_results.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Evaluation Report
    print("\n" + "=" * 70)
    print("                     EMPIRICAL MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"Model Evaluated     : {model_path.name}")
    print(f"Dataset Split       : {split}")
    print(f"Hardware Device     : {device}")
    print("-" * 70)
    print(f"Precision           : {precision:.4f}  (Ratio of true potholes among all model detections)")
    print(f"Recall              : {recall:.4f}  (Ratio of actual potholes successfully detected)")
    print(f"F1-Score            : {f1_score:.4f}  (Harmonic mean of precision & recall)")
    print(f"mAP @ 0.50 IoU      : {map50:.4f}  (Area under Precision-Recall curve at IoU 0.50)")
    print(f"mAP @ 0.50:0.95 IoU : {map50_95:.4f}  (COCO primary benchmark averaged over IoU 0.50:0.95)")
    print("-" * 70)
    print(f"Latency per Frame   : {total_latency_ms:.2f} ms")
    print(f"Calculated FPS      : {measured_fps} frames/sec")
    print("=" * 70)
    print(f"Detailed JSON audit report written to: {report_file}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Pothole Model on validation/test set.")
    parser.add_argument("--model", type=str, default="models/pothole_yolov8n.pt", help="Path to weights file (.pt)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate: val, test")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or 0")
    args = parser.parse_args()

    model_path = (BASE_DIR / args.model).resolve()
    # Fallback to yolov8n.pt if fine-tuned weights are not yet produced
    if not model_path.exists():
        fallback = BASE_DIR / "yolov8n.pt"
        if fallback.exists():
            model_path = fallback
        else:
            model_path = Path("yolov8n.pt")

    yaml_path = (BASE_DIR / args.data).resolve()

    evaluate_model(
        model_path=model_path,
        data_yaml=yaml_path,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
