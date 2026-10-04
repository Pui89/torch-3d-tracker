from __future__ import annotations

from dataclasses import dataclass

import torch

from torch_3d_tracker.config import CameraIntrinsics
from torch_3d_tracker.data import SyntheticSceneGenerator
from torch_3d_tracker.models import DepthEstimator, ObjectTracker, SpatialFeatureExtractor


@dataclass
class TrackerResult:
    depth_map: torch.Tensor
    spatial_features: torch.Tensor
    tracks: list[dict]

    def summary(self) -> str:
        return (
            f"tracks={len(self.tracks)}, "
            f"depth_mean={self.depth_map.mean().item():.3f}, "
            f"feature_shape={tuple(self.spatial_features.shape)}"
        )


class TrackerPipeline:
    def __init__(
        self,
        camera: CameraIntrinsics | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.camera = camera or CameraIntrinsics()
        self.depth_model = DepthEstimator(max_depth=self.camera.max_depth).to(self.device)
        self.spatial_model = SpatialFeatureExtractor(feature_dim=16).to(self.device)
        self.tracker = ObjectTracker()
        self.scene_generator = SyntheticSceneGenerator(self.device)

    def __call__(self, image: torch.Tensor, detections: list[torch.Tensor] | None = None) -> TrackerResult:
        if image.dim() != 4:
            raise ValueError("Expected image tensor with shape [B, C, H, W]")
        image = image.to(self.device)
        depth_map = self.depth_model(image)
        spatial_features = self.spatial_model(image, depth_map)

        if detections is None:
            detections = [torch.tensor([[10.0, 10.0, 30.0, 30.0]], device=self.device)]

        tracks = self.tracker.update(detections)
        return TrackerResult(depth_map=depth_map, spatial_features=spatial_features, tracks=tracks)

    def generate_demo_sequence(self, num_frames: int = 4, height: int = 128, width: int = 128) -> list[torch.Tensor]:
        return self.scene_generator.generate_sequence(num_frames=num_frames, height=height, width=width)
