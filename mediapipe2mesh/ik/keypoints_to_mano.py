"""MediaPipe landmark retargeting and MANO inverse kinematics."""

import numpy as np

from inverse_kinematics.armatures import MANOArmature
from inverse_kinematics.models import KinematicModel
from inverse_kinematics.solver import Solver


N_POSE = 17
MANO_KEYPOINT_PARENTS = np.array([
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14,
    3, 6, 9, 12, 15,
])
FINGER_CHAINS = (
    (1, 2, 3, 16), (4, 5, 6, 17), (7, 8, 9, 18),
    (10, 11, 12, 19), (13, 14, 15, 20),
)


class PoseOnlyWrapper:
    def __init__(self, core, n_pose):
        self.core = core
        self.n_params = n_pose
        self.shape = np.zeros(core.n_shape_params)
        self.global_rotation = np.zeros(3)

    def run(self, pose_pca):
        return self.core.set_params(
            pose_pca=pose_pca,
            pose_glb=self.global_rotation,
            shape=self.shape,
        )[1]


class Keypoints2Mano:
    def __init__(self, model_path='./MANO_RIGHT.npz', side=None,
                 max_iter=5, pose_smoothing=0.70):
        mesh = KinematicModel(model_path, MANOArmature, scale=1000)
        self.side = (side or ('left' if 'LEFT' in model_path.upper()
                              else 'right')).lower()
        if self.side not in ('left', 'right'):
            raise ValueError("side must be 'left' or 'right'")
        self.wrapper = PoseOnlyWrapper(mesh, n_pose=N_POSE)
        self.solver = Solver(
            max_iter=max_iter, verbose=False, central_difference=False
        )
        self.mesh = mesh
        self.rest_wrist = mesh.keypoints[0].copy()
        self.canonical_rest_keypoints = mesh.keypoints - self.rest_wrist
        if self.side == 'left':
            self.canonical_rest_keypoints[:, 0] *= -1.0
        self.bone_lengths = np.zeros(21)
        for joint in range(1, 21):
            parent = MANO_KEYPOINT_PARENTS[joint]
            self.bone_lengths[joint] = np.linalg.norm(
                mesh.keypoints[joint] - mesh.keypoints[parent]
            )
        self.local_to_mano = np.eye(3)
        self.local_to_camera = np.eye(3)
        self.pose_smoothing = float(pose_smoothing)
        self._pose = np.zeros(N_POSE)
        self._has_pose = False

    def get_mano_params(self, keypoints):
        keypoints = self.to_wrist_coordinate(
            np.asarray(keypoints, dtype=np.float64).copy()
        )
        keypoints = self.mediapipe2mano_joints(keypoints)
        keypoints = self.retarget_bone_lengths(keypoints)
        keypoints = self.set_default_rotation(keypoints)
        prior = self._pose if self._has_pose else np.zeros_like(self._pose)
        pose_est = self.solver.solve(
            self.wrapper, keypoints, init=self._pose, u=20, v=5,
            prior=prior, regularization=2.0,
        )
        pose_est = np.clip(pose_est, -3.0, 3.0)
        if self._has_pose:
            alpha = self.pose_smoothing
            pose_est = alpha * pose_est + (1.0 - alpha) * self._pose
        self._pose = pose_est
        self._has_pose = True
        self.mesh.set_params(
            pose_pca=pose_est,
            pose_glb=self.wrapper.global_rotation,
            shape=self.wrapper.shape,
        )
        return pose_est.copy()

    def reset(self):
        self._pose = np.zeros(N_POSE)
        self._has_pose = False

    def to_wrist_coordinate(self, hand_data):
        def normalize(vector):
            return vector / max(np.linalg.norm(vector), 1e-8)
        centered = hand_data - hand_data[0].copy()
        index = centered[5]
        pinky = centered[17]
        normal = normalize(np.cross(pinky, index))
        x_axis = normalize(index)
        y_axis = normal
        z_axis = normalize(np.cross(x_axis, y_axis))
        axes = np.stack((x_axis, y_axis, z_axis), axis=1)
        signs = np.array([1.0, 1.0, -1.0])
        if self.side == 'left':
            signs[1] = -1.0
        camera_to_local = axes @ np.diag(signs)
        self.local_to_camera = camera_to_local.T
        return centered @ camera_to_local

    def get_vertices(self):
        return self.mesh.verts

    def get_faces(self):
        return self.mesh.faces

    def _camera_orient_points(self, points):
        mirror = np.eye(3)
        if self.side == 'left':
            mirror[0, 0] = -1.0
        return points @ mirror @ self.local_to_mano.T @ self.local_to_camera

    def get_camera_oriented_vertices(self):
        return self._camera_orient_points(self.mesh.verts - self.mesh.keypoints[0])

    def get_camera_oriented_keypoints(self):
        return self._camera_orient_points(self.mesh.keypoints - self.mesh.keypoints[0])

    def mediapipe2mano_joints(self, keypoints):
        mapping = [
            0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5,
            6, 17, 10, 11, 12, 19, 7, 8, 9, 18,
        ]
        mano_keypoints = np.zeros((21, 3))
        for index, mano_index in enumerate(mapping):
            mano_keypoints[mano_index] = keypoints[index]
        return mano_keypoints

    def retarget_bone_lengths(self, keypoints):
        retargeted = np.zeros_like(keypoints)
        retargeted[0] = keypoints[0]
        for joint in range(1, 21):
            parent = MANO_KEYPOINT_PARENTS[joint]
            direction = keypoints[joint] - keypoints[parent]
            length = np.linalg.norm(direction)
            if length < 1e-8:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction /= length
            retargeted[joint] = (
                retargeted[parent] + direction * self.bone_lengths[joint]
            )
        return retargeted

    def set_default_rotation(self, keypoints):
        palm_joints = np.array([1, 4, 7, 10])
        source = keypoints[palm_joints] - keypoints[0]
        target = self.canonical_rest_keypoints[palm_joints]
        u, _, vt = np.linalg.svd(source.T @ target)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        self.local_to_mano = rotation
        keypoints = (keypoints - keypoints[0]) @ rotation
        for chain in FINGER_CHAINS:
            mcp = chain[0]
            offset = self.canonical_rest_keypoints[mcp] - keypoints[mcp]
            keypoints[list(chain)] += offset
        if self.side == 'left':
            keypoints[:, 0] *= -1.0
        keypoints += self.rest_wrist - keypoints[0]
        return keypoints
