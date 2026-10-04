from __future__ import annotations

import torch
import torch.nn as nn

from torch_3d_tracker.config import CameraIntrinsics


class DepthEstimator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        depth_logits = self.network(image)
        depth_map = torch.sigmoid(depth_logits) * self.max_depth
        return depth_map


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, feature_dim: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.feature_head = nn.Conv2d(hidden_channels, feature_dim, kernel_size=1)

    def forward(self, image: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        depth_weighted = features * (1.0 + depth_map / depth_map.max().clamp_min(1e-6))
        spatial_features = self.feature_head(depth_weighted)
        return spatial_features


class ObjectTracker:
    def __init__(self):
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    @staticmethod
    def iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h

        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    def _predict(self, track: dict) -> list[float]:
        bbox = list(track["bbox"])
        vel = track["velocity"]
        return [bbox[0] + vel[0], bbox[1] + vel[1], bbox[2] + vel[2], bbox[3] + vel[3]]

    def update(self, detections: list[torch.Tensor]) -> list[dict]:
        flat_detections = []
        for det in detections:
            det = det.detach().cpu().reshape(-1, 4)
            for box in det:
                flat_detections.append(box.tolist())

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        for track_id, track in self.tracks.items():
            prediction = torch.tensor(self._predict(track), dtype=torch.float32)
            best_idx = None
            best_iou = -1.0
            for idx, det in enumerate(flat_detections):
                if idx in matched_detection_indices:
                    continue
                det_tensor = torch.tensor(det, dtype=torch.float32)
                score = self.iou(prediction, det_tensor)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx is not None and best_iou > 0.1:
                matched_track_ids.add(track_id)
                matched_detection_indices.add(best_idx)
                track["bbox"] = flat_detections[best_idx]
                dx = flat_detections[best_idx][0] - track["bbox"][0]
                dy = flat_detections[best_idx][1] - track["bbox"][1]
                track["velocity"] = [dx, dy, dx * 0.1, dy * 0.1]
                track["hits"] += 1
                track["last_seen"] += 1

        for idx, det in enumerate(flat_detections):
            if idx in matched_detection_indices:
                continue
            self.tracks[self.next_id] = {
                "track_id": self.next_id,
                "bbox": det,
                "velocity": [0.0, 0.0, 0.0, 0.0],
                "age": 1,
                "hits": 1,
                "last_seen": 1,
            }
            self.next_id += 1

        result = []
        for track in self.tracks.values():
            result.append({
                "track_id": track["track_id"],
                "bbox": track["bbox"],
                "velocity": track["velocity"],
                "hits": track["hits"],
                "age": track["age"],
            })
        return result
