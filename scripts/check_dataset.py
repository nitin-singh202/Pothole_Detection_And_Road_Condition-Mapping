"""
Dataset Quality Assurance and Sanity Checking Utility.

Validates YOLOv8 dataset integrity, label formatting, coordinate bounds,
detects duplicate/corrupt images, calculates real dataset distribution statistics,
and optionally renders visual annotation overlays for human inspection.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
import yaml

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.utils.logger import setup_logger

logger = setup_logger("check_dataset")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_md5(file_path: Path) -> str:
    """Calculates MD5 hash of an image file to identify exact duplicate files."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetAuditor:
    """Audits YOLOv8 formatted datasets for completeness, correctness, and statistical health."""

    def __init__(self, data_yaml_path: Path, output_vis_dir: Path):
        self.data_yaml_path = data_yaml_path
        self.output_vis_dir = output_vis_dir
        self.config = self._load_yaml()
        self.dataset_root = self._resolve_dataset_root()
        self.classes = self.config.get("names", {0: "pothole"})
        if isinstance(self.classes, list):
            self.classes = {i: name for i, name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def _load_yaml(self) -> dict:
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at: {self.data_yaml_path}")
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_dataset_root(self) -> Path:
        raw_path = self.config.get("path", "")
        if not raw_path:
            return self.data_yaml_path.parent
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = (self.data_yaml_path.parent / resolved).resolve()
        return resolved

    def audit_split(self, split_name: str, rel_img_dir: str, max_visualize: int = 0) -> Dict:
        """Audits an individual dataset split (train, val, or test)."""
        logger.info(f"--- Auditing Split: [{split_name.upper()}] ---")

        img_dir = self.dataset_root / rel_img_dir
        # By YOLO convention, labels folder is at matching depth replacing 'images' with 'labels'
        label_dir_str = str(img_dir).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        if label_dir_str == str(img_dir):
            label_dir = self.dataset_root / "labels" / split_name
        else:
            label_dir = Path(label_dir_str)

        if not img_dir.exists():
            logger.warning(f"Image directory does not exist: {img_dir}")
            return {"exists": False, "split": split_name}

        image_files = [p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        total_images = len(image_files)

        stats = {
            "exists": True,
            "split": split_name,
            "total_images": total_images,
            "valid_images": 0,
            "corrupt_images": 0,
            "duplicate_images": 0,
            "images_with_labels": 0,
            "background_images": 0,  # Images with no annotations (intentional negative samples)
            "missing_label_files": 0,
            "total_bounding_boxes": 0,
            "invalid_labels": 0,
            "class_distribution": {cid: 0 for cid in self.classes.keys()},
            "area_ratios": [],
        }

        if total_images == 0:
            logger.warning(f"Split [{split_name}] contains 0 images.")
            return stats

        seen_md5_hashes: Dict[str, Path] = {}
        visualized_count = 0

        for img_path in image_files:
            # 1. Check Image Readability
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"CORRUPT IMAGE: Unable to decode {img_path.name}")
                stats["corrupt_images"] += 1
                continue

            h, w = img.shape[:2]
            stats["valid_images"] += 1

            # 2. Check for Duplicate Image Hash
            file_hash = compute_file_md5(img_path)
            if file_hash in seen_md5_hashes:
                logger.warning(f"DUPLICATE DETECTED: {img_path.name} is identical to {seen_md5_hashes[file_hash].name}")
                stats["duplicate_images"] += 1
            else:
                seen_md5_hashes[file_hash] = img_path

            # 3. Locate and Validate Corresponding Label File
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                stats["missing_label_files"] += 1
                stats["background_images"] += 1
                continue

            # Read label lines
            with open(label_path, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]

            if len(lines) == 0:
                stats["background_images"] += 1
                continue

            stats["images_with_labels"] += 1
            boxes_to_draw = []

            for line_idx, line in enumerate(lines, start=1):
                tokens = line.split()
                if len(tokens) != 5:
                    logger.error(f"INVALID LABEL: {label_path.name}:L{line_idx} - expected 5 tokens, got {len(tokens)}")
                    stats["invalid_labels"] += 1
                    continue

                try:
                    cls_id = int(tokens[0])
                    xc = float(tokens[1])
                    yc = float(tokens[2])
                    bw = float(tokens[3])
                    bh = float(tokens[4])
                except ValueError:
                    logger.error(f"NON-NUMERIC VALUE in label: {label_path.name}:L{line_idx}")
                    stats["invalid_labels"] += 1
                    continue

                # Validate class id
                if cls_id not in self.classes:
                    logger.error(f"UNKNOWN CLASS ID {cls_id} (Expected one of {list(self.classes.keys())}) in {label_path.name}:L{line_idx}")
                    stats["invalid_labels"] += 1
                    continue

                # Validate normalized bounding box coordinates: 0.0 <= val <= 1.0
                coords = [xc, yc, bw, bh]
                if any(c < 0.0 or c > 1.0 for c in coords) or bw <= 0.0 or bh <= 0.0:
                    logger.error(f"OUT-OF-BOUNDS BBOX in {label_path.name}:L{line_idx} -> ({xc}, {yc}, {bw}, {bh})")
                    stats["invalid_labels"] += 1
                    continue

                # Valid box
                stats["total_bounding_boxes"] += 1
                stats["class_distribution"][cls_id] = stats["class_distribution"].get(cls_id, 0) + 1
                stats["area_ratios"].append(bw * bh)

                if visualized_count < max_visualize:
                    boxes_to_draw.append((cls_id, xc, yc, bw, bh))

            # 4. Optional Visual Overlay Generation
            if boxes_to_draw and visualized_count < max_visualize:
                self._draw_and_save_overlay(img, img_path.name, boxes_to_draw, split_name)
                visualized_count += 1

        return stats

    def _draw_and_save_overlay(self, img: np.ndarray, filename: str, boxes: List[Tuple], split_name: str):
        """Draws bounding boxes and labels onto image for visual inspection."""
        h, w = img.shape[:2]
        canvas = img.copy()

        for cls_id, xc, yc, bw, bh in boxes:
            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            class_name = self.classes.get(cls_id, f"class_{cls_id}")
            color = (0, 0, 255)  # Red in BGR

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_name}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(canvas, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_dir = self.output_vis_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"annotated_{filename}"
        cv2.imwrite(str(out_file), canvas)

    def run_full_audit(self, max_visualize: int = 5) -> bool:
        """Runs audit across all configured dataset splits."""
        splits_to_check = [
            ("train", self.config.get("train", "images/train")),
            ("val", self.config.get("val", "images/val")),
            ("test", self.config.get("test", "images/test")),
        ]

        all_passed = True
        overall_stats = []

        print("\n" + "=" * 70)
        print("               YOLOv8 DATASET AUDIT & HEALTH REPORT")
        print("=" * 70)
        print(f"Dataset Root : {self.dataset_root}")
        print(f"Classes      : {self.classes}")
        print("=" * 70 + "\n")

        for split_name, rel_path in splits_to_check:
            if not rel_path:
                continue
            split_stats = self.audit_split(split_name, rel_path, max_visualize=max_visualize)
            if not split_stats.get("exists", False):
                continue

            overall_stats.append(split_stats)

            print(f"Split: [{split_name.upper()}]")
            print(f"  • Total Images Found     : {split_stats['total_images']}")
            print(f"  • Valid Decoded Images   : {split_stats['valid_images']}")
            print(f"  • Corrupted Images       : {split_stats['corrupt_images']}")
            print(f"  • Duplicate Images       : {split_stats['duplicate_images']}")
            print(f"  • Images with Bounding Box: {split_stats['images_with_labels']}")
            print(f"  • Background/Empty Images: {split_stats['background_images']}")
            print(f"  • Total Pothole Boxes    : {split_stats['total_bounding_boxes']}")
            print(f"  • Invalid Label Entries  : {split_stats['invalid_labels']}")
            
            if split_stats["area_ratios"]:
                avg_area = float(np.mean(split_stats["area_ratios"]))
                min_area = float(np.min(split_stats["area_ratios"]))
                max_area = float(np.max(split_stats["area_ratios"]))
                print(f"  • BBox Area Ratio (Box/Frame) -> Avg: {avg_area:.4f} | Min: {min_area:.4f} | Max: {max_area:.4f}")
            else:
                print("  • BBox Area Ratio        : No annotations found")

            if split_stats["corrupt_images"] > 0 or split_stats["invalid_labels"] > 0:
                all_passed = False
            print("-" * 70)

        if max_visualize > 0 and self.output_vis_dir.exists():
            print(f"\nVisual inspection overlays exported to: {self.output_vis_dir}")

        if all_passed:
            print("\n>>> AUDIT STATUS: [PASSED] - Dataset adheres to YOLOv8 specifications.\n")
        else:
            print("\n>>> AUDIT STATUS: [FAILED] - Corrupt images or invalid labels detected. Review logs.\n")

        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Audit and validate YOLOv8 Pothole Dataset integrity.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml file")
    parser.add_argument("--visualize", type=int, default=3, help="Number of sample annotations to render to outputs/")
    args = parser.parse_args()

    yaml_path = (BASE_DIR / args.data).resolve()
    vis_dir = (BASE_DIR / "outputs/images/dataset_verification").resolve()

    auditor = DatasetAuditor(data_yaml_path=yaml_path, output_vis_dir=vis_dir)
    success = auditor.run_full_audit(max_visualize=args.visualize)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
