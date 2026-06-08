"""
Model configuration and loading utilities for FaceDetector
Author: Sagar
License: MIT
"""

from pathlib import Path
from ultralytics import YOLO


SUPPORTED_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}


def load_model(weights_path: str, device: str = "cpu") -> YOLO:
    """
    Load a YOLO face detection model from weights file.
    Args:
        weights_path: path to .pt weights file
        device: 'cpu' or '0' for GPU
    Returns:
        loaded YOLO model
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model = YOLO(str(path))
    return model


def get_model_info(weights_path: str) -> dict:
    """Return model metadata."""
    model = load_model(weights_path)
    return {
        "weights"   : weights_path,
        "classes"   : model.names,
        "num_classes": len(model.names),
    }