"""
YOLOv8 Pothole Detection Engine.

Wraps the Ultralytics YOLOv8 object detector into a reusable, decoupled class
that executes inference on video frames, filters detections by confidence and IoU,
and returns structured DetectionResult dataclass instances.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO

from app.utils.logger import setup_logger
from config import Config

logger = setup_logger("detector")


@dataclass
class DetectionResult:
    """Structured representation of an individual bounding box detection."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    bbox_width: float
    bbox_height: float
    area_ratio: float
    frame_number: int
    timestamp_sec: float
    severity: str = "LOW"  # Will be enriched by SeverityClassifier

    def to_dict(self) -> dict:
        """Serializes dataclass to a standard Python dictionary."""
        return asdict(self)


class PotholeDetector:
    """Inference engine managing YOLOv8 model lifecycle and per-frame object detection."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        target_size: Optional[int] = None,
    ):
        self.model_path = Path(model_path or Config.MODEL_PATH)
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or Config.IOU_THRESHOLD
        self.device = device or Config.DEVICE
        self.target_size = target_size or Config.TARGET_INFERENCE_SIZE

        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Loads YOLO weights with graceful fallback to yolov8n baseline."""
        target_path = self.model_path
        if not target_path.exists():
            logger.warning(
                f"Configured model weights not found at: {target_path}. "
                f"Falling back to pretrained baseline 'yolov8n.pt'."
            )
            target_path = Path("yolov8n.pt")

        try:
            logger.info(f"Loading YOLOv8 detector weights from: {target_path} (Device: {self.device})")
            model = YOLO(str(target_path))
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {target_path}: {e}")
            raise RuntimeError(f"YOLO Model Initialization Error: {e}")

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_sec: float = 0.0,
    ) -> List[DetectionResult]:
        """
        Executes YOLO inference on a single OpenCV video frame.

        Args:
            frame: Input BGR numpy image array.
            frame_number: Current frame index.
            timestamp_sec: Current video playback timestamp in seconds.

        Returns:
            List[DetectionResult]: List of detected bounding box objects.
        """
        if frame is None or frame.size == 0:
            logger.warning(f"Frame #{frame_number} is empty or None. Skipping detection.")
            return []

        frame_h, frame_w = frame.shape[:2]
        frame_total_area = float(frame_w * frame_h)

        # Run inference via Ultralytics model
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.target_size,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []

        if not results or len(results) == 0:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract tensor coordinates to CPU numpy arrays
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy_arr[i]
            conf = float(conf_arr[i])
            cls_id = int(cls_arr[i])

            bw = float(x2 - x1)
            bh = float(y2 - y1)
            box_area = bw * bh
            area_ratio = box_area / frame_total_area if frame_total_area > 0 else 0.0

            class_name = self.model.names.get(cls_id, f"class_{cls_id}")

            detection = DetectionResult(
                class_id=cls_id,
                class_name=class_name,
                confidence=round(conf, 4),
                x1=round(float(x1), 2),
                y1=round(float(y1), 2),
                x2=round(float(x2), 2),
                y2=round(float(y2), 2),
                bbox_width=round(bw, 2),
                bbox_height=round(bh, 2),
                area_ratio=round(area_ratio, 6),
                frame_number=frame_number,
                timestamp_sec=round(timestamp_sec, 3),
            )
            detections.append(detection)

        return detections
