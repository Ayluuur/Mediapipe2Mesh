"""Temporal filters used by landmark and screen-position tracking."""

import numpy as np


def smoothing_alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Low-lag adaptive filter for arbitrary NumPy arrays."""

    def __init__(self, min_cutoff=1.2, beta=4.0, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.reset()

    def reset(self):
        self.value = None
        self.derivative = None
        self.raw_value = None
        self.timestamp = None

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=np.float64)
        if self.value is None:
            self.value = value.copy()
            self.raw_value = value.copy()
            self.derivative = np.zeros_like(value)
            self.timestamp = timestamp
            return self.value.copy()
        dt = float(np.clip(timestamp - self.timestamp, 1.0 / 120.0, 0.1))
        raw_derivative = (value - self.raw_value) / dt
        derivative_alpha = smoothing_alpha(self.derivative_cutoff, dt)
        self.derivative += derivative_alpha * (raw_derivative - self.derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
        alpha = smoothing_alpha(cutoff, dt)
        self.value += alpha * (value - self.value)
        self.raw_value = value.copy()
        self.timestamp = timestamp
        return self.value.copy()

