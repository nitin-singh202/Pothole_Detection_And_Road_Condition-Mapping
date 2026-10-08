"""
Video Frame Preprocessing and Transformation Utilities.

Handles frame format validation, color-space conversions, aspect-ratio letterboxing,
and normalization required prior to YOLOv8 tensor inference.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    target_size: int = 640,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Prepares a raw OpenCV frame for YOLOv8 inference.

    Args:
        frame: Input BGR image from cv2.VideoCapture.
        target_size: Square dimension for YOLO inference (default 640).

    Returns:
        Tuple[np.ndarray, float, Tuple[int, int]]:
            - Preprocessed image (BGR letterboxed).
            - Scale ratio applied.
            - (pad_w, pad_h) applied for letterboxing.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame: Frame is None or has zero dimensions.")

    letterboxed, ratio, (pad_w, pad_h) = letterbox_image(frame, new_shape=(target_size, target_size))
    return letterboxed, ratio, (pad_w, pad_h)


def letterbox_image(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes and pads image while maintaining original aspect ratio.

    Args:
        im: Input BGR image.
        new_shape: Target (height, width).
        color: Padding fill color in BGR (default gray 114,114,114).
        auto: Minimum rectangle padding.
        scale_fill: Stretch to fit target shape without padding.
        scaleup: If False, only scale down, do not scale up.
        stride: Padding stride constraint.

    Returns:
        Tuple[np.ndarray, float, Tuple[int, int]]: (padded_image, scale_ratio, (pad_w, pad_h))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (int(dw), int(dh))
