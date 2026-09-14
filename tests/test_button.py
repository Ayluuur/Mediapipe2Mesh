"""Button contact, release, sweep and MANO marker regression checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from mediapipe2mesh.interaction.button import DepthButton, mano_index_tip_to_scene


class ButtonTests(unittest.TestCase):
    def setUp(self):
        self.button = DepthButton([110, 30, 370], 90, 60, 12, 5)

    def point(self, x=0, y=0, z=0):
        return self.button.center + [x, y, z]

    def test_press_displaces_and_highlights_then_releases(self):
        button = self.button
        vertices = np.asarray(button.cap.vertices).copy()
        button.update({'left': self.point()})
        self.assertTrue(button.pressed)
        np.testing.assert_allclose(np.asarray(button.cap.vertices), vertices + [0, 0, 12])
        np.testing.assert_allclose(np.asarray(button.cap.vertex_colors)[0], [0.15, 1, 0.35])
        button.update({'left': self.point(z=8)})
        self.assertEqual(button.press_count, 1)
        button.update({'left': self.point(z=-20)})
        self.assertFalse(button.pressed)
        np.testing.assert_allclose(np.asarray(button.cap.vertices), vertices)

    def test_both_hands_hold_until_last_releases(self):
        self.button.update({'left': self.point(), 'right': self.point(x=20)})
        self.assertEqual(self.button.contacts, {'left', 'right'})
        self.button.update({'right': self.point(x=20)})
        self.assertTrue(self.button.pressed)
        self.assertEqual(self.button.press_count, 1)
        self.button.update({})
        self.assertFalse(self.button.pressed)

    def test_depth_and_edges_reject_false_hits(self):
        for point in (self.point(z=-50), self.point(z=50), self.point(x=60),
                      self.point(y=40)):
            self.button.update({})
            self.button.update({'left': point})
            self.assertFalse(self.button.pressed)

    def test_fast_forward_crossing_and_missing_reset(self):
        self.button.update({'left': self.point(z=-30)})
        self.button.update({'left': self.point(z=40)})
        self.assertTrue(self.button.pressed)
        self.button.update({})
        self.button.update({'left': self.point(z=40)})
        self.assertFalse(self.button.pressed)
        self.button.update({'left': self.point(z=-30)})
        self.assertFalse(self.button.pressed)  # Reverse crossing is not a push.

    def test_release_hysteresis_and_invalid_tracking(self):
        self.button.update({'left': self.point(z=-4)})
        self.button.update({'left': self.point(z=-7)})
        self.assertTrue(self.button.pressed)
        self.button.update({'left': [np.nan, 0, 0]})
        self.assertFalse(self.button.pressed)

    def test_markers_removed_on_tracking_loss(self):
        viewer = Mock()
        self.button.update({'left': self.point(), 'right': self.point()})
        self.button.update_geometry(viewer)
        self.assertEqual(self.button.visible_markers, {'left', 'right'})
        self.button.update({})
        self.button.update_geometry(viewer)
        self.assertEqual(viewer.remove_geometry.call_count, 2)

    def test_tip_uses_same_scene_transform_as_mesh(self):
        state = SimpleNamespace(
            keypoints=np.zeros((21, 3)), display_wrist=[0.5, 0.5],
            scene_translation=np.array([10, 20, 200]),
        )
        state.keypoints[16] = [1, 2, 3]
        np.testing.assert_allclose(mano_index_tip_to_scene(state), [11, 18, 197])
        state.keypoints = None
        self.assertIsNone(mano_index_tip_to_scene(state))


if __name__ == '__main__':
    unittest.main()
