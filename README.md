# Torch 3D Tracker

A PyTorch project for 3D object tracking, monocular depth estimation, and spatial feature extraction in a clean object-oriented pipeline.

## Why this project

This project goes beyond typical beginner ML demos by combining:

- A custom object tracker with IoU-based matching and motion smoothing
- A depth-estimation head that predicts dense depth from RGB input
- Spatial feature extraction for 3D-aware scene interpretation
- Object-oriented modules for reuse and extension

## Architecture

The project is organized around a few reusable building blocks:

- `DepthEstimator`: predicts a dense depth map from a single RGB frame
- `SpatialFeatureExtractor`: converts depth + image features into 3D-aware features
- `ObjectTracker`: tracks boxes across frames using IoU matching and velocity state
- `TrackerPipeline`: composes the full sequence and produces a structured result

## Project layout

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── demo.py
├── src
│   └── torch_3d_tracker
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       ├── pipeline.py
│       └── utils.py
└── tests
    └── test_pipeline.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python demo.py
```

This generates a synthetic sequence and prints a compact summary of the tracked objects and depth statistics.

## Example usage

```python
import torch
from torch_3d_tracker.pipeline import TrackerPipeline

pipeline = TrackerPipeline(device=torch.device("cpu"))
image = torch.rand(1, 3, 64, 64)
result = pipeline(image, detections=[torch.tensor([[10.0, 10.0, 32.0, 32.0]])])

print(result.depth_map.shape)
print(result.tracks)
print(result.spatial_features.shape)
```

## Extend this project

Possible next steps:

- Replace the synthetic depth head with a pretrained encoder such as a ResNet backbone
- Add Kalman filtering for smoother multi-object tracking
- Integrate feature matching across frames for temporally consistent tracks
- Add real video input and OpenCV-based preprocessing
- Replace the demo detector with a YOLO-like object detector

## License

MIT
