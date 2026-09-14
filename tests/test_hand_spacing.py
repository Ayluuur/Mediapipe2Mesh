"""Move complete MANO hands in a synthetic camera, including shrinking wrists."""

import unittest
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from mediapipe2mesh.config import PROJECT_ROOT
from mediapipe2mesh.ik import Keypoints2Mano
from mediapipe2mesh.visualization.scene import SceneMapper, configure_view, create_scene_bounds
from mediapipe2mesh.apps.common import update_hand_scene_position
from mediapipe2mesh.tracking.filters import OneEuroFilter
from mediapipe2mesh.tracking.depth import MultiHandDepthEstimator
from mediapipe2mesh.config import load_config
from mediapipe2mesh.interaction import mano_pinch_point_to_scene


MP_TO_MANO = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5,
              6, 17, 10, 11, 12, 19, 7, 8, 9, 18]


class HandSpacingTests(unittest.TestCase):
    def test_sideways_scale_with_different_real_and_mano_palm_proportions(self):
        mapper = SceneMapper()
        actual = np.zeros((21, 3))
        actual[[0, 5, 9, 13, 17]] = [
            [0, 0, 0], [-50, -40, 0], [-15, -50, 0],
            [15, -50, 0], [50, -40, 0],
        ]
        reference = np.zeros((21, 3))
        reference[MP_TO_MANO] = actual * [1, 1.6, 1]
        for sign in (-1, 1):
            corrections = {}
            for angle in (0, 30, 60, 75, 85, 90, 95, 105, 120, 150, 180):
                a = np.deg2rad(angle)
                rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                                     [-np.sin(a), 0, np.cos(a)]])
                world = actual @ rotation.T
                # Keep moving closer/farther while the palm stays on its side.
                for magnification in (1, 0.5, 2):
                    screen = np.zeros((21, 3))
                    screen[:, :2] = (world[:, :2] + [sign * 100, 60]) * magnification / [640, 480] + 0.5
                    old_scale = mapper.palm_scale(screen, world, reference, 640, 480)
                    scale = mapper.palm_scale(screen, world, reference, 640, 480,
                                              segment_corrections=corrections)
                    self.assertIsNotNone(scale)
                    position = mapper.screen_translation(screen, reference, 640, 480,
                                                         75, mm_per_pixel=scale)
                    np.testing.assert_allclose(position, [sign * 100, -60, 75], atol=1e-8)
                    if angle == 75:
                        self.assertGreater(old_scale * magnification, 1.4)
                    if angle in (0, 180):
                        self.assertAlmostEqual(scale, old_scale)

    def test_fast_wrist_rotation_uses_current_detection_not_stale_ik(self):
        mapper = SceneMapper()
        for reference, _ in self.hands:
            state = SimpleNamespace(keypoints=reference.copy(), scene_translation=None,
                                    scene_position_filter=OneEuroFilter())
            depths = MultiHandDepthEstimator(load_config('viewer.json').depth_estimation)
            depths.request_calibration(-3)
            for frame, angle in enumerate((0, 45, 80, 135, 180, 90, 20, 0)):
                a = np.deg2rad(angle)
                rotation = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                                     [0, np.sin(a), np.cos(a)]])
                world = reference[MP_TO_MANO] @ rotation.T
                # Same-frame weak-perspective observation, wrist held fixed.
                screen = np.zeros((21, 3))
                screen[:, :2] = (world[:, :2] + [-100, 80]) / [640, 480] + 0.5
                scale = mapper.palm_scale(screen, world / 1000, reference, 640, 480)
                self.assertIsNotNone(scale)
                self.assertAlmostEqual(scale, 1.0)
                # Deliberately stale and distorted IK must not move the wrist.
                state.keypoints = reference * (0.5 if frame % 2 else 1.5)
                z = depths.update({'left': screen}, frame / 30, 640, 480,
                                  palm_sizes={'left': 20 / scale})['left']
                update_hand_scene_position(state, {'screen': screen}, mapper,
                                           z, 640, 480, frame / 30, mm_per_pixel=scale)
                np.testing.assert_allclose(state.scene_translation,
                                           [-100, -80, depths.estimators['left'].reference_depth])

    def test_palm_back_and_world_size_bias_use_same_metric_scale(self):
        mapper = SceneMapper()
        for reference, _ in self.hands:
            for angle in (0, 30, 60, 120, 150, 180):
                a = np.deg2rad(angle)
                rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                                     [-np.sin(a), 0, np.cos(a)]])
                world = reference[MP_TO_MANO] @ rotation.T
                for magnification in (0.5, 1, 2):
                    screen = np.zeros((21, 3))
                    screen[:, :2] = (world[:, :2] + [100, 60]) * magnification / [640, 480] + 0.5
                    for detector_scale in (0.7, 1.3):
                        scale = mapper.palm_scale(screen, world * detector_scale / 1000,
                                                  reference, 640, 480)
                        self.assertIsNotNone(scale)
                        position = mapper.screen_translation(screen, reference, 640, 480,
                                                             75, mm_per_pixel=scale)
                        np.testing.assert_allclose(position, [100, -60, 75], atol=1e-8)

    def test_unreliable_projection_keeps_scale_and_allows_lateral_motion(self):
        mapper = SceneMapper()
        reference, _ = self.hands[0]
        state = SimpleNamespace(keypoints=reference, scene_translation=None,
                                scene_position_filter=OneEuroFilter())
        screen = np.zeros((21, 3))
        screen[:, :2] = 0.5
        update_hand_scene_position(state, {'screen': screen}, mapper, 75, 640, 480,
                                   0, mm_per_pixel=1)
        self.assertIsNone(mapper.palm_scale(screen, reference[MP_TO_MANO], reference, 640, 480))
        screen[0, 0] += 0.1
        update_hand_scene_position(state, {'screen': screen}, mapper, 75, 640, 480,
                                   1, mm_per_pixel=np.nan)
        self.assertGreater(state.scene_translation[0], 0)
        self.assertEqual(state.scene_mm_per_pixel, 1)

    def test_perspective_palm_flip_has_no_large_spacing_jump(self):
        mapper = SceneMapper()
        for reference, _ in self.hands:
            recovered = []
            for angle in (0, 30, 60, 80, 100, 120, 150, 180):
                a = np.deg2rad(angle)
                rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                                     [-np.sin(a), 0, np.cos(a)]])
                current = reference @ rotation.T
                screen = self.screen(current, 100, 600)
                scale = mapper.palm_scale(screen, current[MP_TO_MANO] / 1000,
                                          reference, 640, 480)
                position = mapper.screen_translation(screen, reference, 640, 480,
                                                     75, mm_per_pixel=scale)
                recovered.append(position[0])
            self.assertLess(np.ptp(recovered), 5)
            self.assertAlmostEqual(recovered[0], recovered[-1], places=6)

    @classmethod
    def setUpClass(cls):
        cls.hands = []
        for side in ('left', 'right'):
            model = Keypoints2Mano(str(PROJECT_ROOT / ('MANO_' + side.upper() + '.npz')), side)
            points = model.mesh.keypoints - model.mesh.keypoints[0]
            across = points[1] - points[7]
            across /= np.linalg.norm(across)
            forward = points[4] - across * np.dot(points[4], across)
            forward /= np.linalg.norm(forward)
            axes = np.stack((across, -forward, np.cross(across, -forward)), axis=1)
            cls.hands.append((points @ axes,
                              (model.mesh.verts - model.mesh.keypoints[0]) @ axes))

    def screen(self, points, x, distance, width=640, height=480):
        camera_points = points[MP_TO_MANO] + [x, 0, distance]
        screen = np.zeros((21, 3))
        screen[:, :2] = (camera_points[:, :2] / camera_points[:, 2:] * 600
                         / [width, height] + 0.5)
        return screen

    def test_whole_mesh_spacing_when_both_hands_approach_and_recede(self):
        mapper = SceneMapper()
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        if not visualizer.create_window(visible=False, width=640, height=480):
            self.skipTest('Open3D unavailable')
        try:
            bounds = create_scene_bounds(visualizer)
            configure_view(visualizer, 'front')
            camera = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
            eye = np.linalg.solve(camera.extrinsic[:3, :3], -camera.extrinsic[:3, 3])
            def project(vertices):
                points = vertices @ camera.extrinsic[:3, :3].T + camera.extrinsic[:3, 3]
                pixels = points @ camera.intrinsic.intrinsic_matrix.T
                return pixels[:, :2] / pixels[:, 2:]
            recovered = []
            old_gaps = []
            for distance, depth in ((300, 300), (600, 150), (1200, 10)):
                scene_hands = []
                old_hands = []
                input_hands = []
                centers = []
                for x, (points, vertices) in zip((-100, 100), self.hands):
                    screen = self.screen(points, x, distance)
                    input_vertices = vertices + [x, 0, distance]
                    input_hands.append(input_vertices[:, 0] / input_vertices[:, 2])
                    # Reproduce the superseded screen-wrist/view-ray mapping.
                    old = mapper.wrist_translation(screen[0], depth)
                    old[:2] = eye[:2] + (depth - eye[2]) / (150 - eye[2]) * (old[:2] - eye[:2])
                    old_hands.append(vertices[:, 0] + old[0])
                    scale = mapper.palm_scale(screen, points[MP_TO_MANO] / 1000, points, 640, 480)
                    translation = mapper.screen_translation(screen, points, 640, 480, depth,
                                                            mm_per_pixel=scale)
                    centers.append(translation[0])
                    self.assertAlmostEqual(translation[0], x, delta=5)
                    scene_hands.append(mapper.hand_vertices(vertices, screen[0], depth,
                                                           translation=translation))
                recovered.append(centers[1] - centers[0])
                self.assertLess(input_hands[0].max(), input_hands[1].min())
                old_gaps.append(old_hands[1].min() - old_hands[0].max())
                # Entire meshes stay separate, not merely their wrist landmarks.
                self.assertLess(scene_hands[0][:, 0].max(), scene_hands[1][:, 0].min())
                projected = sorted((project(hand) for hand in scene_hands),
                                   key=lambda hand: hand[:, 0].mean())
                self.assertLess(projected[0][:, 0].max(), projected[1][:, 0].min())
            self.assertLess(np.ptp(recovered), 5)
            self.assertGreater(old_gaps[0], 0)
            self.assertLess(old_gaps[-1], 0)
        finally:
            visualizer.destroy_window()

    def test_independent_depth_and_real_lateral_motion(self):
        mapper = SceneMapper()
        for distance in (300, 600, 1200):
            for points, _ in self.hands:
                for x in (-120, -40, 40, 120):
                    screen = self.screen(points, x, distance, 1280, 720)
                    scale = mapper.palm_scale(screen, points[MP_TO_MANO] / 1000, points, 1280, 720)
                    translation = mapper.screen_translation(screen, points, 1280, 720, 75,
                                                            mm_per_pixel=scale)
                    self.assertAlmostEqual(translation[0], x, delta=5)
                    self.assertEqual(translation[2], 75)

    def test_world_filter_and_grip_share_mesh_translation(self):
        points, vertices = self.hands[0]
        state = SimpleNamespace(keypoints=points, display_wrist=np.array([0.5, 0.5]),
                                scene_position_filter=OneEuroFilter(), scene_translation=None)
        mapper = SceneMapper()
        for frame, distance in enumerate((300, 600, 1200, 600, 300)):
            screen = self.screen(points, -100, distance)
            update_hand_scene_position(state, {'screen': screen}, mapper, 75, 640, 480, frame / 30)
            self.assertAlmostEqual(state.scene_translation[0], -100, delta=5)
            grip = mano_pinch_point_to_scene(state, mapper, 75)
            expected = ((points[16] + points[20]) / 2 * [1, -1, -1]
                        + state.scene_translation)
            np.testing.assert_allclose(grip, expected)
        before = state.scene_translation.copy()
        update_hand_scene_position(state, {'screen': np.full((21, 3), np.nan)},
                                   mapper, 100, 640, 480, 1)
        np.testing.assert_allclose(state.scene_translation[:2], before[:2])
        self.assertEqual(state.scene_translation[2], 100)


if __name__ == '__main__':
    unittest.main()
