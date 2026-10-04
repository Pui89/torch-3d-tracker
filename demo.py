from __future__ import annotations

import torch

from torch_3d_tracker.config import CameraIntrinsics
from torch_3d_tracker.pipeline import TrackerPipeline


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = TrackerPipeline(camera=CameraIntrinsics(), device=device)

    sequence = pipeline.generate_demo_sequence(num_frames=4, height=128, width=128)

    for idx, image in enumerate(sequence):
        result = pipeline(
            image,
            detections=[
                torch.tensor([[18.0, 22.0, 44.0, 52.0]], device=device),
                torch.tensor([[60.0, 48.0, 94.0, 82.0]], device=device),
            ],
        )
        print(
            f"frame {idx}: "
            f"tracks={len(result.tracks)}, "
            f"depth_mean={result.depth_map.mean().item():.3f}, "
            f"feature_dim={tuple(result.spatial_features.shape)}"
        )


if __name__ == "__main__":
    main()
