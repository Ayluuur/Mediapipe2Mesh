"""Actual MANO assets: FK, mirrored retargeting, bone API and solver regressions."""

import json
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from inverse_kinematics.armatures import MANOArmature
from inverse_kinematics.models import KinematicModel
from inverse_kinematics.solver import Solver
from mediapipe2mesh.config import PROJECT_ROOT, load_config
from mediapipe2mesh.apps.common import extract_detections
from mediapipe2mesh.ik import Keypoints2Mano
from mediapipe2mesh.tracking.hand_state import HandState
from mediapipe2mesh.tracking.handedness import map_handedness


MP_TO_MANO = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5,
              6, 17, 10, 11, 12, 19, 7, 8, 9, 18]


def converter(side='right', mirrored=False, **kwargs):
    return Keypoints2Mano(PROJECT_ROOT / ('MANO_' + side.upper() + '.npz'),
                          side=side, mirrored_input=mirrored,
                          pose_smoothing=1, **kwargs)


def sample(hand, amount=0.7):
    pose = np.zeros((16, 3))
    pose[1:, 2] = amount
    hand.mesh.set_params(pose_abs=pose)
    return hand.mesh.keypoints.copy()


class ModelTests(unittest.TestCase):
    def test_fk_and_pose_correctives_match_independent_skinning(self):
        model = KinematicModel(PROJECT_ROOT / 'MANO_RIGHT.npz', MANOArmature, 1000)
        pose = np.random.default_rng(21).normal(0, .25, (16, 3))
        model.set_params(pose_abs=pose)
        rotations = Rotation.from_rotvec(pose).as_matrix()
        joints = model.J * 1000
        transforms = np.tile(np.eye(4), (16, 1, 1))
        for j in range(16):
            transforms[j, :3, :3] = rotations[j]
            transforms[j, :3, 3] = joints[j] if j == 0 else joints[j] - joints[model.parents[j]]
            if j:
                transforms[j] = transforms[model.parents[j]] @ transforms[j]
        np.testing.assert_allclose(model.keypoints[:16], transforms[:, :3, 3], atol=1e-9)
        feature = (rotations[1:] - np.eye(3)).ravel()
        posed = (model.mesh_template + np.einsum('vcp,p->vc', model.mesh_pose_basis, feature)) * 1000
        expected = np.zeros_like(posed)
        for j in range(16):
            points = ((posed - joints[j]) @ transforms[j, :3, :3].T
                      + transforms[j, :3, 3])
            expected += model.skinning_weights[:, j, None] * points
        np.testing.assert_allclose(model.verts, expected, atol=1e-9)
        full = model.keypoints.copy()
        model.set_params(pose_abs=pose, keypoints_only=True)
        np.testing.assert_allclose(model.keypoints, full, atol=1e-9)


class RetargetTests(unittest.TestCase):
    def test_application_config_selects_anatomical_hand_and_preserves_flexion(self):
        # Exercise the label/model boundary, not just a correctly selected IK.
        for filename in ('interaction.json', 'viewer.json'):
            config = load_config(filename)
            mirrored = config.camera.mirror
            for side in ('left', 'right'):
                with self.subTest(config=filename, side=side):
                    source = converter(side)
                    # The learned MANO mean supplies a natural flexed pose,
                    # rather than assuming an arbitrary axis is flexion.
                    source.mesh.set_params(pose_pca=np.zeros(45))
                    target = source.mesh.keypoints.copy()
                    rest = source.canonical_rest_keypoints.copy()
                    normal = np.cross(rest[1], rest[7])
                    normal /= np.linalg.norm(normal)
                    rotation = Rotation.from_rotvec([.3, -.5, .2]).as_matrix()
                    basis = rotation.T @ np.diag([-1 if mirrored else 1, 1, 1])
                    target = (target - target[0]) @ basis
                    label = side if mirrored else ('left' if side == 'right' else 'right')
                    points = SimpleNamespace(landmark=[SimpleNamespace(x=.5, y=.5)])
                    results = SimpleNamespace(
                        multi_hand_landmarks=[points],
                        multi_hand_world_landmarks=[target[MP_TO_MANO] / 1000],
                        multi_handedness=[SimpleNamespace(classification=[
                            SimpleNamespace(label=label.title(), score=1.)])],
                    )
                    detected = extract_detections(
                        results, config.tracking.handedness_map, mirrored
                    )[0]
                    self.assertEqual(detected['raw_side'], side)
                    hand = converter(detected['raw_side'], mirrored,
                                     max_iter=config.mano.iterations)
                    hand.get_mano_params(detected['world'])
                    actual = hand.get_camera_oriented_keypoints()
                    # Signed finger height relative to the same anatomical
                    # palm normal must stay on the target side of the palm.
                    camera_normal = normal @ basis
                    for mcp, tip in ((1, 16), (4, 17), (7, 18), (10, 19)):
                        expected_height = (target[tip] - target[mcp]) @ camera_normal
                        actual_height = (actual[tip] - actual[mcp]) @ camera_normal
                        self.assertGreater(abs(expected_height), 2.)
                        self.assertGreater(expected_height * actual_height, 0.)

    def test_flat_and_curled_hands_with_rotation_and_mirror(self):
        rotation = Rotation.from_rotvec([.4, -.7, .3]).as_matrix()
        for side in ('left', 'right'):
            for mirrored in (False, True):
                hand = converter(side, mirrored)
                reflection = np.diag([-1., 1, 1]) if mirrored else np.eye(3)
                for amount in (0., .5, 1.):
                    with self.subTest(side=side, mirrored=mirrored, amount=amount):
                        target = sample(hand, amount) @ rotation.T @ reflection
                        hand.reset()
                        params = hand.get_mano_params(target[MP_TO_MANO] / 1000 + [.2, -.1, .4])
                        self.assertEqual(params.shape, (45,))
                        expected = target - target[0]
                        error = np.sqrt(np.mean(np.sum((hand.get_camera_oriented_keypoints() - expected) ** 2, axis=1)))
                        self.assertLess(error, 3.0)
                        if amount == 0:
                            self.assertLess(error, 1e-7)
                        self.assertLess(np.max(np.linalg.norm(hand.mesh.pose, axis=1)), 1.5)
                        skeleton = hand.get_skeleton()
                        np.testing.assert_allclose(skeleton.positions, hand.get_camera_oriented_keypoints()[:16], atol=1e-8)
                        np.testing.assert_allclose(np.linalg.det(skeleton.rotations), 1, atol=1e-8)

    def test_mirror_preserves_pose_and_changes_only_display(self):
        normal = converter()
        mirrored = converter(mirrored=True)
        target = sample(normal)
        normal.reset()
        normal.get_mano_params(target[MP_TO_MANO] / 1000)
        mirrored.get_mano_params(target[MP_TO_MANO] * [-1, 1, 1] / 1000)
        np.testing.assert_allclose(normal.mesh.pose, mirrored.mesh.pose, atol=1e-8)
        np.testing.assert_allclose(mirrored.get_camera_oriented_vertices(), normal.get_camera_oriented_vertices() * [-1, 1, 1], atol=1e-8)
        np.testing.assert_array_equal(mirrored.get_faces(True), normal.get_faces(True)[:, ::-1])
        for side in ('left', 'right'):
            self.assertEqual(map_handedness(side, 'auto', True), side)
            self.assertNotEqual(map_handedness(side, 'auto', False), side)

    def test_invalid_input_keeps_last_valid_snapshot(self):
        hand = converter()
        target = sample(hand)[MP_TO_MANO] / 1000
        hand.get_mano_params(target)
        before = hand.get_skeleton().transforms
        collapsed = target.copy()
        collapsed[2] = collapsed[1]
        collinear = target.copy()
        collinear[17] = collinear[0] + 2 * (collinear[5] - collinear[0])
        for bad in (np.zeros((20, 3)), np.full((21, 3), np.nan), collapsed, collinear):
            with self.assertRaises(ValueError):
                hand.get_mano_params(bad)
            np.testing.assert_array_equal(hand.get_skeleton().transforms, before)
        hand.reset()
        with self.assertRaises(RuntimeError):
            hand.get_skeleton()

    def test_legacy_pca_dimension_can_be_requested(self):
        hand = converter(n_pose=17)
        self.assertEqual(hand.get_mano_params(sample(hand, 0)[MP_TO_MANO]).shape, (17,))


class SkeletonTests(unittest.TestCase):
    def test_hierarchy_quaternions_scene_and_snapshot_ownership(self):
        for mirrored in (False, True):
            hand = converter(mirrored=mirrored)
            target = sample(hand)
            if mirrored:
                target[:, 0] *= -1
            hand.reset()
            hand.get_mano_params(target[MP_TO_MANO] / 1000)
            camera = hand.get_skeleton()
            for data in (camera, hand.get_skeleton('model'), camera.to_scene([20, -30, 100])):
                local = data.local_transforms
                for joint in range(1, 16):
                    np.testing.assert_allclose(data.transforms[data.parents[joint]] @ local[joint], data.transforms[joint], atol=1e-8)
                np.testing.assert_allclose(Rotation.from_quat(data.quaternions).as_matrix(), data.rotations, atol=1e-8)
                np.testing.assert_allclose(Rotation.from_quat(data.local_quaternions).as_matrix(), data.local_rotations, atol=1e-8)
                np.testing.assert_allclose(data.displacements, data.positions - data.rest_positions)
                json.dumps(data.to_dict(), allow_nan=False)
            scene = camera.to_scene([20, -30, 100])
            np.testing.assert_allclose(scene.positions, camera.positions * [1, -1, -1] + [20, -30, 100])
            np.testing.assert_allclose(scene.displacements[0], [20, -30, 100], atol=1e-8)
            camera.transforms[:] = 0
            self.assertGreater(np.linalg.norm(hand.get_skeleton().transforms), 0)

    def test_async_snapshot_and_discarded_frame_reset(self):
        state = HandState('right', 5, 1, mirrored_input=False)
        landmarks = sample(state.converter)[MP_TO_MANO] / 1000
        self.assertIsNone(state.get_skeleton())
        result = state._solve(landmarks)
        state.future = Future()
        state.future.set_result(result)
        state.collect_result()
        np.testing.assert_allclose(state.get_skeleton().positions, state.keypoints[:16])
        state.get_skeleton().transforms[:] = 0
        self.assertGreater(np.linalg.norm(state.get_skeleton().transforms), 0)
        state.future = Future()
        state.begin_track()
        state.future.set_result(result)
        state.collect_result()
        self.assertIsNone(state.get_skeleton())
        self.assertFalse(state.converter._has_pose)


class SolverTests(unittest.TestCase):
    def test_rejects_overshooting_step_and_restores_model(self):
        class Square:
            n_params = 1

            def run(self, params):
                self.params = params.copy()
                return params ** 2

        model = Square()
        initial = np.array([.1])
        result = Solver(max_iter=30).solve(model, np.array([1.]), init=initial)
        np.testing.assert_allclose(result, [1], atol=1e-5)
        np.testing.assert_array_equal(initial, [.1])
        np.testing.assert_array_equal(model.params, result)


if __name__ == '__main__':
    unittest.main()
