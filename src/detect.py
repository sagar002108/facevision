"""
FaceDetector V2
Author: Sagar
License: MIT
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path: str, conf: float = 0.4, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.conf  = conf
        self.device = device

    def detect(self, image: np.ndarray):
        """
        Detect faces in an image.
        Args:
            image: BGR numpy array (OpenCV format)
        Returns:
            list of dicts: [{x1, y1, x2, y2, confidence}]
        """
        results = self.model(image, conf=self.conf,
                             verbose=False, device=self.device)[0]
        faces = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            faces.append({
                "x1"        : int(x1),
                "y1"        : int(y1),
                "x2"        : int(x2),
                "y2"        : int(y2),
                "confidence": float(box.conf[0])
            })
        return faces

    def draw(self, image: np.ndarray, faces: list) -> np.ndarray:
        """Draw bounding boxes on image."""
        img = image.copy()
        for face in faces:
            cv2.rectangle(img,
                          (face["x1"], face["y1"]),
                          (face["x2"], face["y2"]),
                          (0, 255, 100), 2)
            cv2.putText(img,
                        f"face {face['confidence']:.2f}",
                        (face["x1"], face["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 100), 2)
        return img

    def detect_from_file(self, img_path: str) -> tuple:
        """Load image file, detect, return (image_with_boxes, faces)."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        faces = self.detect(img)
        drawn = self.draw(img, faces)
        return drawn, faces


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="FaceDetector V2")
    parser.add_argument("--model",  default="models/face_detector_v2_best.pt")
    parser.add_argument("--image",  default=None, help="Path to image file")
    parser.add_argument("--webcam", action="store_true", help="Run webcam detection")
    parser.add_argument("--conf",   type=float, default=0.4)
    args = parser.parse_args()

    detector = FaceDetector(args.model, conf=args.conf)

    if args.image:
        drawn, faces = detector.detect_from_file(args.image)
        print(f"Detected {len(faces)} face(s)")
        for i, f in enumerate(faces):
            print(f"  Face {i+1}: conf={f['confidence']:.3f} "
                  f"box=({f['x1']},{f['y1']},{f['x2']},{f['y2']})")
        cv2.imwrite("output.jpg", drawn)
        print("Saved → output.jpg")

    elif args.webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Webcam started — press Q to quit")

        frame_count = 0
        fps_start   = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect(frame)
            frame = detector.draw(frame, faces)

            frame_count += 1
            elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS:{fps:.1f} Faces:{len(faces)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2)

            cv2.imshow("FaceDetector V2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()