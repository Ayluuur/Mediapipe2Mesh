"""MediaPipe landmark retargeting and MANO inverse kinematics."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

from inverse_kinematics.armatures import MANOArmature
from inverse_kinematics.models import KinematicModel
from inverse_kinematics.solver import Solver
from inverse_kinematics.parallel import initialize_batch_model, evaluate_batch
from .skeleton import SkeletonPose


N_POSE = 45
MANO_KEYPOINT_PARENTS = np.array([
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14,
    3, 6, 9, 12, 15,
])
FINGER_CHAINS = (
    (1, 2, 3, 16), (4, 5, 6, 17), (7, 8, 9, 18),
    (10, 11, 12, 19), (13, 14, 15, 20),
)


class PoseOnlyWrapper:
    def __init__(self, core, n_pose, jacobian_workers=1, jacobian_backend='thread'):
        self.core = core
        self.n_params = n_pose
        self.shape = np.zeros(core.n_shape_params)
        self.global_rotation = np.zeros(3)
        self.jacobian_workers = jacobian_workers
        self.jacobian_backend = jacobian_backend
        self.worker_pids = set()
        self.worker_stats = {}
        self._process_pools = []
        self._executor = None
        if jacobian_workers == 0:
            self.run_batch = None  # Reference scalar finite-difference path.

    def run_batch(self, poses):
        if self.jacobian_workers == 1:
            return self.core.keypoints_batch(poses, self.shape, self.global_rotation)
        chunks = np.array_split(poses, min(self.jacobian_workers, len(poses)))
        if self.jacobian_backend == 'process':
            if not self._process_pools:
                # A dedicated queue per shard ensures that short batches cannot
                # all be consumed by the first worker that finishes spawning.
                self._process_pools = [ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize_batch_model,
                    initargs=(self.core.model_path, self.core.armature, self.core.scale),
                ) for _ in range(self.jacobian_workers)]
            jobs = [pool.submit(
                evaluate_batch, chunk, self.shape, self.global_rotation
            ) for pool, chunk in zip(self._process_pools, chunks)]
            results = [job.result() for job in jobs]
            for _, pid, cpu_seconds in results:
                self.worker_pids.add(pid)
                stats = self.worker_stats.setdefault(pid, {'batches': 0, 'cpu_seconds': 0.0})
                stats['batches'] += 1
                stats['cpu_seconds'] += cpu_seconds
            return np.concatenate([points for points, _, _ in results])
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.jacobian_workers)
        jobs = [self._executor.submit(
            self.core.keypoints_batch, chunk, self.shape, self.global_rotation
        ) for chunk in chunks]
        return np.concatenate([job.result() for job in jobs])

    def close(self):
        for pool in self._process_pools:
            pool.shutdown(wait=True)
        self._process_pools.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def run(self, pose_pca):
        return self.core.set_params(
            pose_pca=pose_pca,
            pose_glb=self.global_rotation,
            shape=self.shape,
            keypoints_only=True,
        )[1]


class Keypoints2Mano:
    def __init__(self, model_path='./MANO_RIGHT.npz', side=None,
                 max_iter=5, pose_smoothing=0.70, n_pose=N_POSE,
                 mirrored_input=True, jacobian_workers=1, jacobian_backend='thread'):
        model_path = str(model_path)
        if not 1 <= n_pose <= 45:
            raise ValueError('n_pose must be between 1 and 45')
        if not 0.0 <= pose_smoothing <= 1.0:
            raise ValueError('pose_smoothing must be between 0 and 1')
        if (isinstance(jacobian_workers, bool)
                or not isinstance(jacobian_workers, int)
                or not 0 <= jacobian_workers <= 45):
            raise ValueError('jacobian_workers must be an integer in [0, 45]')
        if jacobian_backend not in ('thread', 'process'):
            raise ValueError('jacobian_backend must be thread or process')
        mesh = KinematicModel(model_path, MANOArmature, scale=1000)
        self.side = (side or ('left' if 'LEFT' in model_path.upper()
                              else 'right')).lower()
        if self.side not in ('left', 'right'):
            raise ValueError("side must be 'left' or 'right'")
        self.wrapper = PoseOnlyWrapper(mesh, n_pose=n_pose,
                                      jacobian_workers=jacobian_workers,
                                      jacobian_backend=jacobian_backend)
        self.mirrored_input = bool(mirrored_input)
        self._input_basis = np.diag([-1.0, 1.0, 1.0]) if mirrored_input else np.eye(3)
        self.solver = Solver(
            max_iter=max_iter, verbose=False, central_difference=False
        )
        self.mesh = mesh
        self.rest_wrist = mesh.keypoints[0].copy()
        self.canonical_rest_keypoints = mesh.keypoints - self.rest_wrist
        self.bone_lengths = np.zeros(21)
        for joint in range(1, 21):
            parent = MANO_KEYPOINT_PARENTS[joint]
            self.bone_lengths[joint] = np.linalg.norm(
                mesh.keypoints[joint] - mesh.keypoints[parent]
            )
        self.local_to_mano = np.eye(3)
        self.local_to_camera = np.eye(3)
        self.pose_smoothing = float(pose_smoothing)
        # Zero PCA is the learned mean (a bent hand), not the flat bind pose.
        self._neutral_pose = np.linalg.lstsq(
            mesh.pose_pca_basis[:n_pose].T, -mesh.pose_pca_mean, rcond=None
        )[0]
        self._pose_encoder = np.linalg.pinv(mesh.pose_pca_basis[:n_pose].T)
        self._pose = self._neutral_pose.copy()
        self._has_pose = False
        self.fit_error_mm = None

    def get_mano_params(self, keypoints):
        keypoints = np.asarray(keypoints, dtype=np.float64)
        if keypoints.shape != (21, 3) or not np.all(np.isfinite(keypoints)):
            raise ValueError('keypoints must be a finite (21, 3) world-landmark array')
        # Reject missing bones rather than inventing directions that twist fingers.
        ordered = self.mediapipe2mano_joints(keypoints)
        lengths = np.linalg.norm(ordered[1:] - ordered[MANO_KEYPOINT_PARENTS[1:]], axis=1)
        if np.any(lengths < 1e-8):
            raise ValueError('keypoints contain a zero-length bone')
        keypoints = self.to_wrist_coordinate(
            keypoints @ self._input_basis
        )
        keypoints = self.mediapipe2mano_joints(keypoints)
        keypoints = self.retarget_bone_lengths(keypoints)
        keypoints = self.set_default_rotation(keypoints)
        # A persistent flat-pose prior regularizes unobserved axial twist;
        # the previous solution is only a warm start, not a moving twist target.
        prior = self._neutral_pose
        seed = self._direction_seed(keypoints)
        def cost(pose):
            return (np.sum((self.wrapper.run(pose) - keypoints) ** 2)
                    + 2.0 * np.sum((pose - prior) ** 2))
        initial = seed if cost(seed) < cost(self._pose) else self._pose
        pose_est = self.solver.solve(
            self.wrapper, keypoints, init=initial, u=20, v=5,
            prior=prior, regularization=2.0,
        )
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
        self.fit_error_mm = float(np.sqrt(np.mean(np.sum(
            (self.mesh.keypoints - keypoints) ** 2, axis=1
        ))))
        return pose_est.copy()

    def _direction_seed(self, target):
        """Fit each outgoing bone with minimal local swing, without axial twist."""
        rotations = np.tile(np.eye(3), (16, 1, 1))
        local = rotations.copy()
        rest = self.canonical_rest_keypoints
        for chain in FINGER_CHAINS:
            for joint, child in zip(chain[:-1], chain[1:]):
                parent = MANO_KEYPOINT_PARENTS[joint]
                a = rest[child] - rest[joint]
                b = rotations[parent].T @ (target[child] - target[joint])
                a = a / np.linalg.norm(a)
                b = b / np.linalg.norm(b)
                axis = np.cross(a, b)
                sine = np.linalg.norm(axis)
                cosine = np.clip(a @ b, -1.0, 1.0)
                if sine > 1e-8:
                    rotvec = axis / sine * np.arctan2(sine, cosine)
                elif cosine < 0:
                    basis = np.eye(3)[np.argmin(np.abs(a))]
                    axis = np.cross(a, basis)
                    rotvec = axis / np.linalg.norm(axis) * np.pi
                else:
                    rotvec = np.zeros(3)
                local[joint] = Rotation.from_rotvec(rotvec).as_matrix()
                rotations[joint] = rotations[parent] @ local[joint]
        pose = Rotation.from_matrix(local[1:]).as_rotvec().ravel()
        return self._pose_encoder @ (pose - self.mesh.pose_pca_mean)

    def close(self):
        self.wrapper.close()

    def reset(self):
        self._pose = self._neutral_pose.copy()
        self._has_pose = False
        self.fit_error_mm = None
        self.local_to_mano = np.eye(3)
        self.local_to_camera = np.eye(3)
        self.mesh.set_params(pose_abs=np.zeros((16, 3)))

    def to_wrist_coordinate(self, hand_data):
        def normalize(vector):
            return vector / max(np.linalg.norm(vector), 1e-8)
        centered = hand_data - hand_data[0].copy()
        index = centered[5]
        pinky = centered[17]
        if (np.linalg.norm(np.cross(pinky, index))
                < 1e-4 * np.linalg.norm(pinky) * np.linalg.norm(index)):
            raise ValueError('palm landmarks are collinear')
        normal = normalize(np.cross(pinky, index))
        x_axis = normalize(index)
        y_axis = normal
        z_axis = normalize(np.cross(x_axis, y_axis))
        axes = np.stack((x_axis, y_axis, z_axis), axis=1)
        camera_to_local = axes
        self.local_to_camera = camera_to_local.T
        return centered @ camera_to_local

    def get_vertices(self):
        return self.mesh.verts

    def get_faces(self, camera_oriented=False):
        if camera_oriented and self.mirrored_input:
            return self.mesh.faces[:, ::-1].copy()
        return self.mesh.faces.copy()

    def get_skeleton(self, space='camera'):
        """Return a detached snapshot of the same FK bones used by the mesh.

        Camera space is wrist-centered, x right / y down / z away, in mm.
        Model space retains MANO's bind origin. No camera translation can be
        recovered from MediaPipe world landmarks alone.
        """
        if space not in ('camera', 'model'):
            raise ValueError("space must be 'camera' or 'model'")
        if not self._has_pose:
            raise RuntimeError('solve a valid landmark frame before reading skeleton data')
        transforms = self.mesh.joint_transforms.copy()
        rest = self.mesh.J.copy() * self.mesh.scale
        if space == 'camera':
            basis = self._input_basis @ (self.local_to_mano.T @ self.local_to_camera).T
            # A mirrored display also mirrors each joint's local basis. This
            # keeps rotations in SO(3), rather than converting a reflection
            # (det=-1) into an invalid quaternion.
            transforms[:, :3, :3] = basis @ transforms[:, :3, :3] @ self._input_basis
            transforms[:, :3, 3] = (transforms[:, :3, 3] - self.mesh.keypoints[0]) @ basis.T
            rest = (rest - rest[0]) @ basis.T
        parents = self.mesh.parents.astype(np.int64).copy()
        parents[0] = -1
        return SkeletonPose(
            self.side, space, tuple(self.mesh.armature.labels[:16]), parents,
            transforms, rest, self.fit_error_mm,
            mirrored_local_axes=(space == 'camera' and self.mirrored_input),
        )

    def _camera_orient_points(self, points):
        return points @ self.local_to_mano.T @ self.local_to_camera @ self._input_basis

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
        keypoints += self.rest_wrist - keypoints[0]
        return keypoints
