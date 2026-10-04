from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float = 700.0
    fy: float = 700.0
    cx: float = 64.0
    cy: float = 64.0
    max_depth: float = 10.0
