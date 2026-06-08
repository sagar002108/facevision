"""
Training script for FaceDetector
Author: Sagar
License: MIT

Usage:
    python src/train.py --data data/face_dataset.yaml --epochs 30
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml: str, epochs: int = 30, imgsz: int = 640,
          batch: int = 16, device: str = "0", name: str = "face_detector"):

    model = YOLO("yolov8n.pt")

    results = model.train(
        data     = data_yaml,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        patience = 10,
        device   = device,
        save     = True,
        plots    = True,
        workers  = 2,
        name     = name,
        project  = "models"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FaceDetector")
    parser.add_argument("--data",   default="data/face_dataset.yaml")
    parser.add_argument("--epochs", type=int,   default=30)
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--batch",  type=int,   default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name",   default="face_detector_v2")
    args = parser.parse_args()

    print(f"Training FaceDetector")
    print(f"  Data  : {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")

    train(
        data_yaml = args.data,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        name      = args.name
    )