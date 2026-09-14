"""Temporal filters used by landmark and screen-position tracking."""

import numpy as np
from collections import deque


def smoothing_alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Low-lag adaptive filter for arbitrary NumPy arrays."""

    def __init__(self, min_cutoff=1.2, beta=4.0, derivative_cutoff=1.0,
                 median_window=1, max_speed=None):
        if (not np.isfinite(min_cutoff) or min_cutoff <= 0
                or not np.isfinite(beta) or beta < 0
                or not np.isfinite(derivative_cutoff) or derivative_cutoff <= 0):
            raise ValueError('Filter cutoffs must be positive and beta nonnegative')
        if (not isinstance(median_window, int) or median_window < 1
                or median_window % 2 != 1):
            raise ValueError('median_window must be a positive odd integer')
        if max_speed is not None and (not np.isfinite(max_speed) or max_speed <= 0):
            raise ValueError('max_speed must be finite and positive')
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.median_window = median_window
        self.max_speed = max_speed
        self.reset()

    def reset(self):
        self.value = None
        self.derivative = None
        self.raw_value = None
        self.timestamp = None
        self.samples = deque(maxlen=self.median_window)

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            if self.value is None:
                raise ValueError('Cannot initialize filter with nonfinite values')
            return self.value.copy()
        if self.value is None:
            self.samples.extend(value.copy() for _ in range(self.median_window))
            self.value = value.copy()
            self.raw_value = value.copy()
            self.derivative = np.zeros_like(value)
            self.timestamp = timestamp
            return self.value.copy()
        dt = float(np.clip(timestamp - self.timestamp, 1.0 / 120.0, 0.1))
        self.samples.append(value.copy())
        if self.median_window > 1:
            value = np.median(np.stack(self.samples), axis=0)
        if self.max_speed is not None:
            # Bound the innovation BEFORE computing the adaptive cutoff.
            # Otherwise a detector spike opens the filter and passes through.
            delta = value - self.value
            distance = np.linalg.norm(delta)
            limit = self.max_speed * dt
            if distance > limit:
                value = self.value + delta * (limit / distance)
        raw_derivative = (value - self.raw_value) / dt
        derivative_alpha = smoothing_alpha(self.derivative_cutoff, dt)
        self.derivative += derivative_alpha * (raw_derivative - self.derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
        alpha = smoothing_alpha(cutoff, dt)
        self.value += alpha * (value - self.value)
        self.raw_value = value.copy()
        self.timestamp = timestamp
        return self.value.copy()
