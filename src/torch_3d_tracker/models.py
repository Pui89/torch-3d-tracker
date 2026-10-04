from __future__ import annotations

import numpy as np
import torch


class SyntheticSceneGenerator:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")

    def generate_scene(self, height: int = 128, width: int = 128) -> torch.Tensor:
        y, x = np.mgrid[0:height, 0:width]
        base = np.zeros((height, width), dtype=np.float32)
        base += 0.35 + 0.2 * np.sin(x / 12.0)
        base += 0.25 * np.cos(y / 18.0)

        depth_map = np.clip(base + 0.6, 0.0, 1.0)
        rgb = np.stack(
            [
                depth_map,
                0.6 * np.ones_like(depth_map),
                0.8 * np.ones_like(depth_map),
            ],
            axis=0,
        )
        image = torch.tensor(rgb, dtype=torch.float32, device=self.device).unsqueeze(0)
        return image

    def generate_sequence(
        self,
        num_frames: int = 4,
        height: int = 128,
        width: int = 128,
    ) -> list[torch.Tensor]:
        sequence: list[torch.Tensor] = []
        for frame_index in range(num_frames):
            base = self.generate_scene(height, width)
            shift = float(frame_index) * 2.0
            if frame_index % 2 == 0:
                base = self._add_object(base, (20.0 + shift, 28.0), (40.0 + shift, 56.0), 0.9)
            else:
                base = self._add_object(base, (50.0, 38.0 + shift * 0.2), (88.0, 80.0 + shift * 0.2), 0.8)
            sequence.append(base)
        return sequence

    def _add_object(
        self,
        image: torch.Tensor,
        top_left: tuple[float, float],
        bottom_right: tuple[float, float],
        intensity: float,
    ) -> torch.Tensor:
        _, h, w = image.shape
        x0 = int(max(0, top_left[0]))
        y0 = int(max(0, top_left[1]))
        x1 = int(min(w, bottom_right[0]))
        y1 = int(min(h, bottom_right[1]))

        obj = image[:, y0:y1, x0:x1].clone()
        obj[0] = torch.clamp(obj[0] + intensity, 0.0, 1.0)
        obj[1] = torch.clamp(obj[1] + 0.2, 0.0, 1.0)
        obj[2] = torch.clamp(obj[2] + 0.1, 0.0, 1.0)
        image[:, y0:y1, x0:x1] = obj
        return image
