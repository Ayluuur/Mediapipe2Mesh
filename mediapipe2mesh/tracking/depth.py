"""Approximate monocular hand depth from apparent palm size."""

import numpy as np

from .filters import OneEuroFilter


PALM_SIZE_SEGMENTS = (
    (0, 9),    # wrist to middle MCP
    (5, 17),   # index MCP to pinky MCP
    (5, 9),
    (9, 13),
    (13, 17),
)


class RelativeDepthEstimator:
    """Estimate positive-toward-user scene Z from apparent palm size."""

    def __init__(self, reference_depth=50.0, minimum_depth=10.0,
                 maximum_depth=300.0, calibration_frames=30,
                 min_cutoff=1.0, beta=0.02, max_speed=500.0,
                 motion_gain=1.5):
        self.reference_depth = float(reference_depth)
        self.minimum_depth = float(minimum_depth)
        self.maximum_depth = float(maximum_depth)
        self.calibration_frames = int(calibration_frames)
        self.max_speed = float(max_speed)
        self.motion_gain = float(motion_gain)
        self.filter = OneEuroFilter(
            min_cutoff=min_cutoff,
            beta=beta,
            derivative_cutoff=1.0,
        )
        self.samples = []
        self.reference_size = None
        self.depth = self.reference_depth
        self.last_seen = None

    @classmethod
    def from_config(cls, config, fallback_reference=None):
        reference = config.reference_depth
        if reference is None:
            reference = fallback_reference
        if reference is None:
            raise ValueError('depth_estimation.reference_depth is required')
        return cls(
            reference_depth=reference,
            minimum_depth=config.minimum_depth,
            maximum_depth=config.maximum_depth,
            calibration_frames=config.calibration_frames,
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            max_speed=config.max_speed,
            motion_gain=config.motion_gain,
        )

    @property
    def calibrated(self):
        return self.reference_size is not None

    @property
    def calibration_progress(self):
        if self.calibrated:
            return 1.0
        return min(len(self.samples) / max(self.calibration_frames, 1), 1.0)

    def update(self, screen_landmarks, timestamp, frame_width, frame_height):
        size = self.measure_palm_size(
            screen_landmarks, frame_width, frame_height
        )
        if size < 5.0:
            return self.depth
        if self.last_seen is None or timestamp - self.last_seen > 0.35:
            self.filter.reset()
        previous_time = self.last_seen
        self.last_seen = timestamp

        if not self.calibrated:
            self.samples.append(size)
            if len(self.samples) >= self.calibration_frames:
                self.reference_size = float(np.median(self.samples))
            raw_depth = self.reference_depth
        else:
            # This project's +Z points toward the user: a larger projected
            # palm means the hand is closer and therefore has a larger Z.
            base_depth = self.reference_depth * size / self.reference_size
            raw_depth = self.reference_depth + (
                base_depth - self.reference_depth
            ) * self.motion_gain
        raw_depth = float(np.clip(
            raw_depth, self.minimum_depth, self.maximum_depth
        ))
        if previous_time is not None and self.max_speed > 0.0:
            dt = float(np.clip(timestamp - previous_time, 1.0 / 120.0, 0.1))
            limit = self.max_speed * dt
            raw_depth = float(np.clip(
                raw_depth, self.depth - limit, self.depth + limit
            ))
        self.depth = float(self.filter(raw_depth, timestamp))
        return self.depth

    @staticmethod
    def measure_palm_size(screen_landmarks, frame_width, frame_height):
        if hasattr(screen_landmarks, 'landmark'):
            points = np.array([
                [point.x * frame_width, point.y * frame_height]
                for point in screen_landmarks.landmark
            ])
        else:
            points = np.asarray(screen_landmarks, dtype=np.float64)[:, :2].copy()
            points[:, 0] *= frame_width
            points[:, 1] *= frame_height
        lengths = [
            np.linalg.norm(points[first] - points[second])
            for first, second in PALM_SIZE_SEGMENTS
        ]
        return float(np.median(lengths))
