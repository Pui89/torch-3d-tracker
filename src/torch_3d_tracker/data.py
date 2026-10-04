from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackState:
    track_id: int
    bbox: list[float]
    velocity: list[float]
    age: int = 0
    hits: int = 1
    last_seen: int = 0

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "velocity": self.velocity,
            "age": self.age,
            "hits": self.hits,
            "last_seen": self.last_seen,
        }
