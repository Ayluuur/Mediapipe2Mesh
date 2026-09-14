"""Synthetic detector spikes, real motion and brief tracking loss."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from mediapipe2mesh.apps.common import retire_missing_single_hand, update_hand_scene_position
from mediapipe2mesh.config import load_config
from mediapipe2mesh.tracking.filters import OneEuroFilter
from mediapipe2mesh.tracking.depth import RelativeDepthEstimator
from mediapipe2mesh.visualization import SceneMapper


class TrackingJitterTests(unittest.TestCase):
    def position_filter(self):
        return OneEuroFilter(**load_config('interaction.json').tracking.position_filter.as_dict())

    def test_isolated_spike_does_not_open_adaptive_filter(self):
        filt = self.position_filter()
        old = OneEuroFilter(min_cutoff=1.5, beta=0.02)
        inputs = [np.zeros(2)] * 10 + [np.array([250., -200.])] + [np.zeros(2)] * 10
        actual = [filt(point, i / 30) for i, point in enumerate(inputs)]
        previous = [old(point, i / 30) for i, point in enumerate(inputs)]
        np.testing.assert_allclose(actual, 0)
        self.assertGreater(np.max(np.linalg.norm(previous, axis=1)), 100)

    def test_sustained_jump_is_bounded_and_reaches_new_position(self):
        filt = self.position_filter()
        positions = [filt([0., 0.], 0)]
        for frame in range(1, 181):
            positions.append(filt([200., 100.], frame / 30))
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 30
        self.assertLessEqual(max(speeds), 500 + 1e-8)
        np.testing.assert_allclose(positions[-1], [200, 100], atol=0.1)

    def test_ordinary_motion_still_follows(self):
        filt = self.position_filter()
        for frame in range(91):
            target = np.array([frame * 2., 0.])  # 60 mm/s.
            actual = filt(target, frame / 30)
        self.assertLess(np.linalg.norm(target - actual), 12)

    def test_metric_scale_spike_is_filtered_before_mesh_translation(self):
        state = SimpleNamespace(keypoints=np.zeros((21, 3)),
                                scene_position_filter=self.position_filter(),
                                scene_translation=None)
        screen = np.full((21, 3), 0.5)
        screen[0, 0] = 0.7
        positions = []
        for frame, scale in enumerate([1.] * 10 + [3.] + [1.] * 10):
            update_hand_scene_position(state, {'screen': screen}, SceneMapper(),
                                       200, 640, 480, frame / 30, mm_per_pixel=scale)
            positions.append(state.scene_translation.copy())
        np.testing.assert_allclose(positions, np.tile([128, 0, 200], (21, 1)))

    def test_depth_spike_does_not_move_hand(self):
        depth = RelativeDepthEstimator(reference_depth=200)
        depth.reference_size = 20
        outputs = [depth.update_size(size, frame / 30)
                   for frame, size in enumerate([20] * 10 + [80] + [20] * 10)]
        np.testing.assert_allclose(outputs, 200)

    def test_short_loss_preserves_track_but_long_loss_retires(self):
        states = {
            'left': SimpleNamespace(last_seen=1., deactivate=Mock()),
            'right': SimpleNamespace(last_seen=1.1, deactivate=Mock()),
        }
        release = Mock()
        retire_missing_single_hand([{}], {'right': {}}, states, release)
        states['left'].deactivate.assert_not_called()
        release.assert_not_called()
        states['right'].last_seen = 1.4
        retire_missing_single_hand([{}], {'right': {}}, states)
        states['left'].deactivate.assert_called_once()

    def test_invalid_sample_does_not_poison_filter_and_reset_clears_median(self):
        filt = self.position_filter()
        filt([0., 0.], 0)
        np.testing.assert_allclose(filt([np.nan, 1.], 1 / 30), [0, 0])
        filt.reset()
        np.testing.assert_allclose(filt([100., 100.], 1), [100, 100])


if __name__ == '__main__':
    unittest.main()
