"""Per-hand asynchronous MANO solve state."""

from copy import deepcopy
from collections import deque
import time

import numpy as np
import open3d as o3d

from mediapipe2mesh.config import PROJECT_ROOT
from mediapipe2mesh.ik import Keypoints2Mano
from .filters import OneEuroFilter
from .performance import SolveSnapshot


HAND_COLORS = {
    'left': (0.35, 0.65, 1.0),
    'right': (1.0, 0.65, 0.35),
}


class HandState:
    def __init__(self, side, max_iter, pose_smoothing, position_filter_config=None,
                 mirrored_input=True, jacobian_workers=1, jacobian_backend='thread'):
        self.side = side
        model_name = 'MANO_LEFT.npz' if side == 'left' else 'MANO_RIGHT.npz'
        self.solver_options = dict(
            model_path=str(PROJECT_ROOT / model_name), side=side,
            max_iter=max_iter, pose_smoothing=pose_smoothing,
            mirrored_input=mirrored_input,
            jacobian_workers=jacobian_workers,
            jacobian_backend=jacobian_backend,
        )
        self.track_generation = 0
        self.result_version = 0
        self.ik_timings = deque(maxlen=20)
        self.converter = Keypoints2Mano(
            str(PROJECT_ROOT / model_name), side, max_iter, pose_smoothing,
            mirrored_input=mirrored_input,
            jacobian_workers=jacobian_workers,
            jacobian_backend=jacobian_backend,
        )
        self.filter = OneEuroFilter(median_window=3)
        position_options = dict(min_cutoff=3.0, beta=0.02,
                                derivative_cutoff=2.0,
                                median_window=3, max_speed=1500.0)
        if position_filter_config is not None:
            position_options.update(position_filter_config.as_dict())
        # The fallback wrist uses normalized image coordinates, whereas the
        # recovered scene position uses millimeters.
        screen_options = dict(position_options)
        screen_options['max_speed'] /= 300.0
        screen_options['beta'] *= 300.0
        self.position_filter = OneEuroFilter(
            **screen_options
        )
        self.scene_reference_keypoints = self.converter.mesh.keypoints.copy()
        # Palm proportions persist across temporary loss/re-entry, like depth
        # calibration. A sideways re-entry must not discard the frontal scale.
        self.scene_segment_corrections = {}
        self.scene_position_filter = OneEuroFilter(**position_options)
        self.scene_translation = None
        self.scene_mm_per_pixel = None
        self.future = None
        self.discard_future = False
        self.pending_landmarks = None
        self.vertices = None
        self.keypoints = None
        self.skeleton = None
        self.last_seen = -np.inf
        self.screen_wrist = None
        self.display_wrist = None
        self.side_candidate = None
        self.side_candidate_frames = 0
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.triangles = o3d.utility.Vector3iVector(
            self.converter.get_faces(camera_oriented=True)
        )
        self.mesh.paint_uniform_color(HAND_COLORS[side])
        self.added_to_viewer = False

    def submit_latest(self, executor):
        if self.future is None and self.pending_landmarks is not None:
            landmarks = self.pending_landmarks
            self.pending_landmarks = None
            if hasattr(executor, 'submit_hand'):
                self.future = executor.submit_hand(self, landmarks)
            else:
                self.future = executor.submit(self._solve, landmarks)

    def collect_result(self):
        if self.future is None or not self.future.done():
            return
        try:
            snapshot = self.future.result()
            vertices, keypoints, skeleton = snapshot
            if not self.discard_future:
                self.vertices = vertices
                self.keypoints = keypoints
                self.skeleton = skeleton
                self.result_version += 1
                seconds = getattr(snapshot, 'ik_seconds', 0.0)
                if np.isfinite(seconds) and seconds > 0:
                    self.ik_timings.append(seconds)
        except Exception as exc:
            print('{} hand IK failed: {}'.format(self.side, exc))
        self.future = None
        if self.discard_future:
            self.converter.reset()
        self.discard_future = False

    def _solve(self, landmarks):
        started = time.perf_counter()
        self.converter.get_mano_params(landmarks)
        seconds = time.perf_counter() - started
        return SolveSnapshot(
            self.converter.get_camera_oriented_vertices().copy(),
            self.converter.get_camera_oriented_keypoints().copy(),
            self.converter.get_skeleton(),
            seconds,
        )

    @property
    def mean_ik_seconds(self):
        return sum(self.ik_timings) / len(self.ik_timings) if self.ik_timings else 0.0

    def get_skeleton(self, space='camera'):
        """Read the last collected mesh frame without touching the IK worker.

        Returns None before a result or after track loss. Scene coordinates
        require scene_translation from the application's position mapper.
        """
        if space not in ('camera', 'scene'):
            raise ValueError("space must be 'camera' or 'scene'")
        if self.skeleton is None:
            return None
        if space == 'scene':
            if self.scene_translation is None:
                return None
            return self.skeleton.to_scene(self.scene_translation)
        return deepcopy(self.skeleton)

    def begin_track(self):
        self.ik_timings.clear()
        self.track_generation += 1
        self.scene_mm_per_pixel = None
        self.scene_position_filter.reset()
        self.scene_translation = None
        self.filter.reset()
        self.position_filter.reset()
        self.vertices = None
        self.keypoints = None
        self.skeleton = None
        if self.future is None:
            self.converter.reset()
        else:
            self.discard_future = True

    def deactivate(self):
        self.ik_timings.clear()
        self.scene_mm_per_pixel = None
        self.scene_position_filter.reset()
        self.scene_translation = None
        self.last_seen = -np.inf
        self.screen_wrist = None
        self.display_wrist = None
        self.pending_landmarks = None
        self.vertices = None
        self.keypoints = None
        self.skeleton = None
        self.side_candidate = None
        self.side_candidate_frames = 0
        self.filter.reset()
        self.position_filter.reset()
        if self.future is not None:
            self.discard_future = True
