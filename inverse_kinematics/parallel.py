"""Spawn-safe workers for independent MANO finite-difference batches."""

import os
import time

from .models import KinematicModel


_model = None


def initialize_batch_model(model_path, armature, scale):
    global _model
    _model = KinematicModel(model_path, armature, scale)


def evaluate_batch(poses, shape, global_rotation):
    started = time.process_time()
    points = _model.keypoints_batch(poses, shape, global_rotation)
    return points, os.getpid(), time.process_time() - started
