"""Per-hand asynchronous MANO solve state."""

import numpy as np
import open3d as o3d

from mediapipe2mesh.config import PROJECT_ROOT
from mediapipe2mesh.ik import Keypoints2Mano
from .filters import OneEuroFilter


HAND_COLORS = {
    'left': (0.35, 0.65, 1.0),
    'right': (1.0, 0.65, 0.35),
}


class HandState:
    def __init__(self, side, max_iter, pose_smoothing):
        self.side = side
        model_name = 'MANO_LEFT.npz' if side == 'left' else 'MANO_RIGHT.npz'
        self.converter = Keypoints2Mano(
            str(PROJECT_ROOT / model_name), side, max_iter, pose_smoothing
        )
        self.filter = OneEuroFilter()
        self.position_filter = OneEuroFilter(
            min_cutoff=1.5, beta=2.0, derivative_cutoff=1.0
        )
        self.scene_reference_keypoints = self.converter.mesh.keypoints.copy()
        # Palm proportions persist across temporary loss/re-entry, like depth
        # calibration. A sideways re-entry must not discard the frontal scale.
        self.scene_segment_corrections = {}
        self.scene_position_filter = OneEuroFilter(min_cutoff=1.5, beta=0.02)
        self.scene_translation = None
        self.scene_mm_per_pixel = None
        self.future = None
        self.discard_future = False
        self.pending_landmarks = None
        self.vertices = None
        self.keypoints = None
        self.last_seen = -np.inf
        self.screen_wrist = None
        self.display_wrist = None
        self.side_candidate = None
        self.side_candidate_frames = 0
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.triangles = o3d.utility.Vector3iVector(
            self.converter.get_faces()
        )
        self.mesh.paint_uniform_color(HAND_COLORS[side])
        self.added_to_viewer = False

    def submit_latest(self, executor):
        if self.future is None and self.pending_landmarks is not None:
            landmarks = self.pending_landmarks
            self.pending_landmarks = None
            self.future = executor.submit(self._solve, landmarks)

    def collect_result(self):
        if self.future is None or not self.future.done():
            return
        try:
            vertices, keypoints = self.future.result()
            if not self.discard_future:
                self.vertices = vertices
                self.keypoints = keypoints
        except Exception as exc:
            print('{} hand IK failed: {}'.format(self.side, exc))
        self.future = None
        self.discard_future = False

    def _solve(self, landmarks):
        self.converter.get_mano_params(landmarks)
        return (
            self.converter.get_camera_oriented_vertices().copy(),
            self.converter.get_camera_oriented_keypoints().copy(),
        )

    def begin_track(self):
        self.scene_mm_per_pixel = None
        self.scene_position_filter.reset()
        self.scene_translation = None
        self.filter.reset()
        self.position_filter.reset()
        self.vertices = None
        self.keypoints = None
        if self.future is None:
            self.converter.reset()
        else:
            self.discard_future = True

    def deactivate(self):
        self.scene_mm_per_pixel = None
        self.scene_position_filter.reset()
        self.scene_translation = None
        self.last_seen = -np.inf
        self.screen_wrist = None
        self.display_wrist = None
        self.pending_landmarks = None
        self.vertices = None
        self.keypoints = None
        self.side_candidate = None
        self.side_candidate_frames = 0
        self.filter.reset()
        self.position_filter.reset()
        if self.future is not None:
            self.discard_future = True

