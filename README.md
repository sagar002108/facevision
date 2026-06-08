# FaceDetector V2

A custom human face detection model trained from scratch on WIDER FACE dataset.
Built as a research project — designed to be the foundation for face recognition,
expression detection, and security systems.

## Results

| Metric    | Score  |
|-----------|--------|
| mAP50     | 0.9183 |
| Precision | 0.8909 |
| Recall    | 0.8474 |
| FPS       | ~10 (CPU) |

## Features
- Detects human faces in images and live webcam
- Trained on 11,457 images with 40,010 face annotations
- Hard negative mining — animals and objects not falsely detected
- Lightweight YOLOv8n backbone — runs on CPU

## Quick Start

### Install
```bash
git clone https://github.com/YOUR_USERNAME/face-detector.git
cd face-detector
pip install -r requirements.txt
```

### Detect faces in an image
```bash
python src/detect.py --image your_photo.jpg
```

### Live webcam detection
```bash
python src/detect.py --webcam
```

### Use as a Python module
```python
from src.detect import FaceDetector

detector = FaceDetector("models/face_detector_v2_best.pt")
faces = detector.detect(image)  # pass OpenCV BGR image
```

## Training

Dataset: [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
See `notebooks/` for full training pipeline.

## Roadmap
- [x] Face Detection V1
- [x] Face Detection V2 (hard negative mining)
- [ ] Facial Landmark Detection
- [ ] Face Recognition
- [ ] Expression Recognition

## License
MIT