"""
Dataset utilities for FaceDetector
Author: Sagar
License: MIT
"""

import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """
    YOLO-format face dataset loader.
    Expects:
        images/ - .jpg files
        labels/ - .txt files (YOLO format: class cx cy w h)
    """

    def __init__(self, img_dir: str, label_dir: str, img_size: int = 640):
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size  = img_size
        self.images    = sorted(self.img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        boxes = []
        if lbl_path.exists() and lbl_path.read_text().strip():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    boxes.append(parts)

        return img, boxes, str(img_path)

    @staticmethod
    def collate_fn(batch):
        imgs, boxes, paths = zip(*batch)
        return list(imgs), list(boxes), list(paths)


def load_image(img_path: str, img_size: int = 640):
    """Load and preprocess single image for inference."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")
    original = img.copy()
    resized  = cv2.resize(img, (img_size, img_size))
    return original, resized