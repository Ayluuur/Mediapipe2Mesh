"""Lightweight IK result shared by thread and process workers."""

from dataclasses import dataclass


@dataclass
class SolveSnapshot:
    vertices: object
    keypoints: object
    skeleton: object
    ik_seconds: float

    def __iter__(self):
        # Preserve existing three-value snapshot consumers.
        return iter((self.vertices, self.keypoints, self.skeleton))
