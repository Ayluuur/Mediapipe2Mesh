"""Independent scalar references for batched and threaded IK derivatives."""

import unittest

import numpy as np

from mediapipe2mesh.config import PROJECT_ROOT
from mediapipe2mesh.ik import Keypoints2Mano


class BatchedIKTests(unittest.TestCase):
    def hand(self, side='right', **options):
        hand = Keypoints2Mano(
            str(PROJECT_ROOT / ('MANO_' + side.upper() + '.npz')),
            side=side, **options,
        )
        self.addCleanup(hand.close)
        return hand

    def test_batch_matches_scalar_with_shape_rotation_and_reduced_pca(self):
        rng = np.random.default_rng(13)
        for side in ('left', 'right'):
            for n_pose in (17, 45):
                for workers in (1, 4):
                    with self.subTest(side=side, n_pose=n_pose, workers=workers):
                        hand = self.hand(side, n_pose=n_pose, jacobian_workers=workers)
                        wrapper = hand.wrapper
                        wrapper.shape[:] = rng.normal(0, .2, wrapper.shape.shape)
                        wrapper.global_rotation[:] = [.1, -.2, .3]
                        poses = rng.normal(0, .4, (7, n_pose))
                        expected = np.stack([wrapper.run(pose) for pose in poses])
                        before = (hand.mesh.pose.copy(), hand.mesh.keypoints.copy(),
                                  hand.mesh.verts.copy(), hand.mesh.joint_transforms.copy())
                        actual = wrapper.run_batch(poses)
                        np.testing.assert_allclose(actual, expected, atol=1e-10)
                        for value, original in zip(
                                (hand.mesh.pose, hand.mesh.keypoints, hand.mesh.verts,
                                 hand.mesh.joint_transforms), before):
                            np.testing.assert_array_equal(value, original)

    def test_solver_matches_scalar_over_motion_and_reset(self):
        mapping = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5,
                   6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
        for central in (False, True):
            for side in ('left', 'right'):
                for mirrored in (False, True):
                    source = self.hand(side)
                    hands = [self.hand(side, max_iter=5, mirrored_input=mirrored,
                                       jacobian_workers=w) for w in (0, 1, 4)]
                    for hand in hands:
                        hand.solver.central_difference = central
                    for bend in (.2, .8, .4):
                        pose = np.zeros((16, 3))
                        pose[1:, 2] = bend
                        source.mesh.set_params(pose_abs=pose)
                        landmarks = source.mesh.keypoints[mapping].copy() / 1000
                        if bend == .4:
                            for hand in hands:
                                hand.reset()
                        for hand in hands:
                            hand.get_mano_params(landmarks)
                        for hand in hands[1:]:
                            np.testing.assert_allclose(
                                hand.get_camera_oriented_vertices(),
                                hands[0].get_camera_oriented_vertices(), atol=1e-5,
                            )
                            self.assertAlmostEqual(hand.fit_error_mm,
                                                   hands[0].fit_error_mm, places=5)

    def test_invalid_worker_counts(self):
        for workers in (-1, 46, True, 1.5, '4'):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.hand(jacobian_workers=workers)

    def test_process_batch_matches_scalar_and_closes_workers(self):
        import multiprocessing

        hand = self.hand(jacobian_workers=8, jacobian_backend='process')
        poses = np.random.default_rng(22).normal(0, .3, (45, 45))
        # Subsequent calls reuse worker models but must honor changed shape/rotation.
        for shape in (0.0, .2):
            hand.wrapper.shape[:] = shape
            hand.wrapper.global_rotation[:] = shape
            expected = np.stack([hand.wrapper.run(pose) for pose in poses])
            actual = hand.wrapper.run_batch(poses)
            np.testing.assert_allclose(actual, expected, atol=1e-10)
            self.assertEqual(len(hand.wrapper.worker_pids), 8)
        pids = hand.wrapper.worker_pids.copy()
        for stats in hand.wrapper.worker_stats.values():
            self.assertEqual(stats['batches'], 2)
            self.assertGreaterEqual(stats['cpu_seconds'], 0)
        hand.close()
        self.assertFalse(pids & {child.pid for child in multiprocessing.active_children()})

    def test_invalid_jacobian_backend(self):
        with self.assertRaises(ValueError):
            self.hand(jacobian_backend='invalid')


if __name__ == '__main__':
    unittest.main()
