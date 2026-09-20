"""Real spawn-process parity and bounded detection prefetch regressions."""

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from mediapipe2mesh.apps.parallel import DetectionPipeline, HandExecutor
from mediapipe2mesh.config import PROJECT_ROOT
from mediapipe2mesh.ik import Keypoints2Mano


class ParallelTests(unittest.TestCase):
    def test_spawn_matches_sequential_and_resets_on_reacquisition(self):
        mapping = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5,
                   6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
        states = []
        references = []
        for side in ('left', 'right'):
            options = dict(
                model_path=str(PROJECT_ROOT / ('MANO_' + side.upper() + '.npz')),
                side=side, max_iter=1, pose_smoothing=.7, mirrored_input=False,
                jacobian_workers=4,
                jacobian_backend='process',
            )
            references.append(Keypoints2Mano(**dict(options, jacobian_workers=0)))
            states.append(SimpleNamespace(
                side=side, solver_options=options, track_generation=0,
            ))
        with HandExecutor() as executor:
            for generation, bend in ((0, .2), (0, .8), (1, .4)):
                jobs = []
                for state, reference in zip(states, references):
                    pose = np.zeros((16, 3))
                    pose[1:, 2] = bend
                    reference.mesh.set_params(pose_abs=pose)
                    landmarks = reference.mesh.keypoints[mapping].copy() / 1000
                    state.track_generation = generation
                    jobs.append(executor.submit_hand(state, landmarks))
                    if generation:
                        reference.reset()
                    reference.get_mano_params(landmarks)
                for job, reference in zip(jobs, references):
                    snapshot = job.result(timeout=90)
                    self.assertGreater(snapshot.ik_seconds, 0)
                    vertices, keypoints, skeleton = snapshot
                    np.testing.assert_allclose(
                        vertices, reference.get_camera_oriented_vertices(), atol=1e-7
                    )
                    np.testing.assert_allclose(
                        keypoints, reference.get_camera_oriented_keypoints(), atol=1e-7
                    )
                    self.assertEqual(skeleton.side, reference.side)
            reports = executor.worker_reports()
            for side, report in reports.items():
                self.assertEqual(len(report['jacobian_pids']), 4)
                self.assertTrue(all(s['batches'] >= 3 for s in report['workers'].values()))
                self.assertNotIn(report['coordinator_pid'], report['jacobian_pids'])
            self.assertFalse(set(reports['left']['jacobian_pids'])
                             & set(reports['right']['jacobian_pids']))

    def test_detection_prefetch_is_bounded_and_keeps_frame_pairs(self):
        calls = []
        prefetched = threading.Event()

        class Camera:
            def read(self):
                index = len(calls)
                calls.append(index)
                return True, np.full((2, 2, 3), index, dtype=np.uint8)

        class Detector:
            def process(self, rgb):
                if int(rgb[0, 0, 0]) == 1:
                    prefetched.set()
                return int(rgb[0, 0, 0])

        pipeline = DetectionPipeline(Camera(), Detector(), False)
        try:
            frame, timestamp, result = pipeline.read()
            self.assertEqual(result, int(frame[0, 0, 0]))
            self.assertTrue(prefetched.wait(5))
            self.assertEqual(calls, [0, 1])
            next_frame, next_timestamp, next_result = pipeline.read()
            self.assertEqual(next_result, 1)
            self.assertEqual(next_result, int(next_frame[0, 0, 0]))
            self.assertGreaterEqual(next_timestamp, timestamp)
        finally:
            pipeline.close()

    def test_detection_failure_is_propagated(self):
        class Camera:
            def read(self):
                raise RuntimeError('camera failure')

        pipeline = DetectionPipeline(Camera(), None, False)
        try:
            with self.assertRaisesRegex(RuntimeError, 'camera failure'):
                pipeline.read()
        finally:
            pipeline.close()


if __name__ == '__main__':
    unittest.main()
