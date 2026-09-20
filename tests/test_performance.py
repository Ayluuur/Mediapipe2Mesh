"""Computation speed, displayed updates and capture throughput stay distinct."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
import numpy as np

from mediapipe2mesh.tracking.hand_state import HandState
from mediapipe2mesh.tracking.performance import FrameRates, SolveSnapshot


class PerformanceTests(unittest.TestCase):
    def test_average_duration_is_used_and_discarded_results_are_excluded(self):
        state = HandState('left', 1, .7)
        for duration in (.01, .03):
            state.future = Future()
            state.future.set_result(SolveSnapshot(np.zeros((778, 3)), np.zeros((21, 3)), None, duration))
            state.collect_result()
        self.assertAlmostEqual(state.mean_ik_seconds, .02)
        self.assertEqual(state.result_version, 2)
        state.collect_result()
        self.assertEqual(state.result_version, 2)
        state.future = Future()
        state.begin_track()
        state.future.set_result(SolveSnapshot(None, None, None, 1.0))
        state.collect_result()
        self.assertEqual(state.mean_ik_seconds, 0)
        self.assertEqual(state.result_version, 2)
        self.assertIsNone(state.vertices)

    def test_timing_window_is_bounded_and_loss_clears_it(self):
        state = HandState('right', 1, .7)
        for n in range(25):
            state.future = Future()
            state.future.set_result(SolveSnapshot(None, None, None, (n + 1) / 1000))
            state.collect_result()
        self.assertEqual(len(state.ik_timings), 20)
        self.assertAlmostEqual(state.mean_ik_seconds, .0155)
        state.deactivate()
        self.assertEqual(state.mean_ik_seconds, 0)

    def test_fast_ik_with_30fps_capture_and_repeated_mesh(self):
        states = {side: SimpleNamespace(result_version=0, mean_ik_seconds=.008,
                  last_seen=1., vertices=object()) for side in ('left', 'right')}
        rates = FrameRates(0.)
        for n in range(1, 31):
            states['left'].result_version += 1
            if n % 3 == 0:
                states['right'].result_version += 1
            rates.update(states, ('left', 'right'), n / 30)
        self.assertAlmostEqual(rates.capture_fps, 30)
        self.assertAlmostEqual(rates.mano_fps['left'], 30)
        self.assertAlmostEqual(rates.mano_fps['right'], 10)
        self.assertIn('125.0 (8.0ms)', rates.labels(states, 1.)[0])
        rates.record_phases(.002, .010, .004)
        self.assertEqual(
            rates.labels(states, 1.)[2],
            'Time ms Capture:2.0 MediaPipe:10.0 Main/render:4.0',
        )
        for n in range(31, 46):
            rates.update(states, (), n / 30)
        self.assertEqual(rates.mano_fps, dict(left=0., right=0.))
        self.assertEqual(rates.labels(states, 1.5)[0], 'IK compute FPS L:-- R:--')


if __name__ == '__main__':
    unittest.main()
