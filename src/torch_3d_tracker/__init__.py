from __future__ import annotations

import torch


def normalize_depth(depth_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    depth_map = depth_map.clamp_min(eps)
    return depth_map / depth_map.max().clamp_min(eps)


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union
