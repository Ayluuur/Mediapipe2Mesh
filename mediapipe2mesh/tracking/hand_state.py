"""Per-hand asynchronous MANO solve state."""

from copy import deepcopy

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
    def __init__(self, side, max_iter, pose_smoothing, position_filter_config=None,
                 mirrored_input=True):
        self.side = side
        model_name = 'MANO_LEFT.npz' if side == 'left' else 'MANO_RIGHT.npz'
        self.converter = Keypoints2Mano(
            str(PROJECT_ROOT / model_name), side, max_iter, pose_smoothing,
            mirrored_input=mirrored_input,
        )
        self.filter = OneEuroFilter(median_window=3)
        position_options = dict(min_cutoff=1.5, beta=0.005,
                                median_window=3, max_speed=500.0)
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
            self.future = executor.submit(self._solve, landmarks)

    def collect_result(self):
        if self.future is None or not self.future.done():
            return
        try:
            vertices, keypoints, skeleton = self.future.result()
            if not self.discard_future:
                self.vertices = vertices
                self.keypoints = keypoints
                self.skeleton = skeleton
        except Exception as exc:
            print('{} hand IK failed: {}'.format(self.side, exc))
        self.future = None
        if self.discard_future:
            self.converter.reset()
        self.discard_future = False

    def _solve(self, landmarks):
        self.converter.get_mano_params(landmarks)
        return (
            self.converter.get_camera_oriented_vertices().copy(),
            self.converter.get_camera_oriented_keypoints().copy(),
            self.converter.get_skeleton(),
        )

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
