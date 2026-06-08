"""
Inference utilities for FaceDetector
Author: Sagar
License: MIT
"""

import cv2
import numpy as np
from src.detect import FaceDetector


def run_webcam(model_path: str, conf: float = 0.4):
    """Run real-time face detection on webcam."""
    detector = FaceDetector(model_path, conf=conf, device="cpu")
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
        fps     = frame_count / elapsed

        cv2.putText(frame, f"FPS:{fps:.1f} Faces:{len(faces)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 255), 2)
        cv2.imshow("FaceDetector V2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {fps:.1f}")


def run_image(model_path: str, img_path: str,
              conf: float = 0.4, save_path: str = "output.jpg"):
    """Run detection on a single image file."""
    detector = FaceDetector(model_path, conf=conf)
    drawn, faces = detector.detect_from_file(img_path)

    print(f"Detected {len(faces)} face(s)")
    for i, f in enumerate(faces):
        print(f"  Face {i+1}: conf={f['confidence']:.3f} "
              f"box=({f['x1']},{f['y1']},{f['x2']},{f['y2']})")

    cv2.imwrite(save_path, drawn)
    print(f"Saved → {save_path}")
    return faces