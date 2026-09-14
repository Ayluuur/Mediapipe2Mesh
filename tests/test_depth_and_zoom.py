"""Regression checks for shared depth calibration and camera zoom."""

import unittest
from unittest.mock import Mock

import numpy as np
import open3d as o3d

from mediapipe2mesh.config import PROJECT_ROOT, load_config
from mediapipe2mesh.ik import Keypoints2Mano
from mediapipe2mesh.tracking.depth import MultiHandDepthEstimator
from mediapipe2mesh.visualization.scene import (
    SceneMapper, configure_view, create_scene_bounds, scroll_zoom,
)


def palm(scale=1.0):
    points = np.zeros((21, 3))
    points[[0, 5, 9, 13, 17], :2] = [
        [0, 0], [-0.06, -0.10], [-0.02, -0.12],
        [0.02, -0.11], [0.06, -0.09],
    ]
    points[:, :2] = points[:, :2] * scale + 0.5
    return points


class DepthTests(unittest.TestCase):
    def tracker(self):
        return MultiHandDepthEstimator(load_config('viewer.json').depth_estimation)

    def update(self, tracker, frame, **scales):
        return tracker.update({side: palm(size) for side, size in scales.items()},
                              frame / 30.0, 640, 480)

    def test_different_initial_sizes_converge_to_same_depth(self):
        tracker = self.tracker()
        for frame in range(30):
            self.update(tracker, frame, left=0.7, right=1.3)
        for frame in range(30, 330):
            depths = self.update(tracker, frame, left=1.1, right=1.1)
        self.assertAlmostEqual(depths['left'], depths['right'], places=5)
        self.assertGreater(depths['left'], tracker.estimators['left'].reference_depth)

    def test_late_hand_uses_existing_reference(self):
        tracker = self.tracker()
        for frame in range(30):
            self.update(tracker, frame, left=0.8)
        reference = tracker.reference_size
        for frame in range(30, 330):
            depths = self.update(tracker, frame, left=1.2, right=1.2)
        self.assertEqual(reference, tracker.reference_size)
        self.assertAlmostEqual(depths['left'], depths['right'], places=5)
        # Genuine size differences still yield independent depths.
        for frame in range(330, 630):
            depths = self.update(tracker, frame, left=0.8, right=1.2)
        self.assertGreater(depths['right'], depths['left'])

    def test_calibration_is_frame_based_and_order_independent(self):
        first, second = self.tracker(), self.tracker()
        for frame in range(30):
            a = self.update(first, frame, left=0.7, right=1.3)
            b = self.update(second, frame, right=1.3, left=0.7)
            self.assertEqual(a, b)
        self.assertEqual(len(first.samples), 30)
        self.assertEqual(first.reference_size, second.reference_size)

    def test_missing_or_invalid_palms_do_not_calibrate(self):
        tracker = self.tracker()
        self.assertEqual(tracker.update({}, 0, 640, 480), {})
        tracker.update({'left': np.full((21, 3), np.nan)}, 1, 640, 480)
        self.update(tracker, 60, right=0)
        self.assertEqual(tracker.samples, [])
        self.assertIsNone(tracker.reference_size)

    def test_manual_calibration_uses_hands_at_deadline_and_resets_filters(self):
        tracker = self.tracker()
        for frame in range(60):
            self.update(tracker, frame, left=0.8, right=1.2)
        old_reference = tracker.reference_size
        tracker.request_calibration(2.0)
        self.update(tracker, 149, left=1.4, right=1.7)
        self.assertEqual(tracker.reference_size, old_reference)
        self.assertIsNone(tracker.calibration_completed_at)
        depths = self.update(tracker, 150, left=1.1, right=1.3)
        for side in depths:
            self.assertAlmostEqual(depths[side], tracker.estimators[side].reference_depth)
        self.assertIsNone(tracker.calibration_deadline)
        self.assertEqual(tracker.calibration_completed_at, 5.0)
        # Equal proportional motion preserves the calibrated common plane.
        for frame in range(151, 300):
            depths = self.update(tracker, frame, left=1.21, right=1.43)
        self.assertAlmostEqual(depths['left'], depths['right'])
        self.assertGreater(depths['left'], tracker.estimators['left'].reference_depth)

    def test_manual_calibration_waits_for_valid_hands(self):
        tracker = self.tracker()
        tracker.request_calibration(0)
        tracker.update({}, 3, 640, 480)
        tracker.update({'left': np.full((21, 3), np.nan)}, 4, 640, 480)
        self.assertEqual(tracker.calibration_deadline, 3)
        self.assertIn('waiting', tracker.calibration_status(4))
        depths = self.update(tracker, 150, left=1.2)
        self.assertAlmostEqual(depths['left'], tracker.estimators['left'].reference_depth)
        self.assertTrue(tracker.estimators['right'].calibrated)
        self.assertIn('complete', tracker.calibration_status(5))

    def test_pressing_enter_again_restarts_countdown(self):
        tracker = self.tracker()
        tracker.request_calibration(0)
        tracker.request_calibration(2)
        self.update(tracker, 90, left=1)
        self.assertEqual(tracker.calibration_deadline, 5)
        self.assertIsNone(tracker.calibration_completed_at)
        self.update(tracker, 150, left=1)
        self.assertEqual(tracker.calibration_completed_at, 5)


class GeometryTests(unittest.TestCase):
    def test_scroll_zoom_in_observer_views(self):
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        if not visualizer.create_window(visible=False, width=640, height=480):
            self.skipTest('Open3D window unavailable')
        try:
            bounds = create_scene_bounds(visualizer)
            mapper = SceneMapper()

            def project(points):
                camera = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
                camera_points = (points @ camera.extrinsic[:3, :3].T
                                 + camera.extrinsic[:3, 3])
                self.assertTrue(np.all(camera_points[:, 2] > 0))
                pixels = camera_points @ camera.intrinsic.intrinsic_matrix.T
                return pixels[:, :2] / pixels[:, 2:]

            offsets = np.array([[-20, 0, 0], [20, 0, 0]], dtype=float)
            # Wheel behavior is verified using rendered size, not scale's name.
            for mode in ('front', 'depth'):
                configure_view(visualizer, mode)
                points = mapper.hand_vertices(offsets, [0.5, 0.5], 150)
                before = np.linalg.norm(np.diff(project(points), axis=0))
                scroll_zoom(visualizer, 0, 1)
                after = np.linalg.norm(np.diff(project(points), axis=0))
                self.assertGreater(after, before)
                scroll_zoom(visualizer, 0, -1)
                restored = np.linalg.norm(np.diff(project(points), axis=0))
                self.assertAlmostEqual(restored, before)
        finally:
            visualizer.destroy_window()

    def test_posed_wrist_is_scene_anchor_for_both_hands(self):
        mapper = SceneMapper()
        for side in ('left', 'right'):
            converter = Keypoints2Mano(
                str(PROJECT_ROOT / ('MANO_' + side.upper() + '.npz')), side)
            for pose in (np.zeros(17), np.linspace(-1, 1, 17)):
                converter.mesh.set_params(pose_pca=pose)
                points = converter.get_camera_oriented_keypoints()
                vertices = converter.get_camera_oriented_vertices()
                np.testing.assert_allclose(points[0], np.zeros(3), atol=1e-10)
                # Mesh vertices and grip landmarks retain their relative offsets.
                np.testing.assert_allclose(vertices[0] - points[4],
                    converter._camera_orient_points(
                        converter.mesh.verts[0] - converter.mesh.keypoints[4]))
                scene = mapper.hand_keypoints(points, [0.4, 0.6], 75)
                np.testing.assert_allclose(scene[0], [-30, -24, 75])

    def test_scroll_reverses_both_directions_and_ignores_horizontal(self):
        visualizer = Mock()
        for delta in (-2, 0, 0.5, 3):
            self.assertFalse(scroll_zoom(visualizer, 10, delta))
            visualizer.get_view_control().scale.assert_called_with(-delta)


if __name__ == '__main__':
    unittest.main()
