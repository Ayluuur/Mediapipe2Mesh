"""Virtual ball, pinch state machine, and one/two-hand manipulation."""

import numpy as np
import open3d as o3d

from mediapipe2mesh.visualization import CAMERA_TO_OPEN3D, SceneMapper


PINCH_COLORS = {'left': (0.2, 0.8, 1.0), 'right': (1.0, 0.55, 0.2)}


def normalize(vector):
    length = np.linalg.norm(vector)
    return np.zeros(3) if length < 1e-8 else vector / length


def rotation_between_vectors(source, target):
    source = normalize(source)
    target = normalize(target)
    if not source.any() or not target.any():
        return np.eye(3)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-7:
        return np.eye(3)
    if cosine < -1.0 + 1e-7:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(source, candidate))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def palm_frame(world_landmarks):
    wrist = world_landmarks[0]
    across = normalize(world_landmarks[5] - world_landmarks[17])
    forward = normalize(world_landmarks[9] - wrist)
    normal = normalize(np.cross(across, forward))
    if not normal.any():
        return np.eye(3)
    across = normalize(np.cross(forward, normal))
    camera_frame = np.stack((across, forward, normal), axis=1)
    return np.diag(CAMERA_TO_OPEN3D) @ camera_frame


def mano_pinch_point_to_scene(hand_state, mapper=None, depth=0.0):
    if hand_state.keypoints is None or hand_state.display_wrist is None:
        return None
    mapper = mapper or SceneMapper()
    points = mapper.hand_keypoints(
        hand_state.keypoints, hand_state.display_wrist, depth,
        translation=hand_state.scene_translation,
    )
    return (points[16] + points[20]) * 0.5


class InteractiveBall:
    def __init__(self, radius=35.0, depth=35.0):
        self.base_radius = float(radius)
        self.center = np.array([0.0, 0.0, float(depth)])
        self.rotation = np.eye(3)
        self.scale = 1.0
        self.mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.base_radius, resolution=28
        )
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color((0.95, 0.72, 0.15))
        self.base_vertices = np.asarray(self.mesh.vertices).copy()
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.base_radius * 1.45, origin=(0.0, 0.0, 0.0)
        )
        self.frame_base_vertices = np.asarray(self.frame.vertices).copy()

    @property
    def radius(self):
        return self.base_radius * self.scale

    def contains(self, point, tolerance=6.0):
        return np.linalg.norm(point - self.center) <= self.radius + tolerance

    def add_to(self, visualizer):
        visualizer.add_geometry(self.mesh, reset_bounding_box=False)
        visualizer.add_geometry(self.frame, reset_bounding_box=False)
        self.update_geometry(visualizer)

    def update_geometry(self, visualizer):
        vertices = (self.base_vertices * self.scale) @ self.rotation.T
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices + self.center)
        self.mesh.compute_vertex_normals()
        frame_vertices = (
            (self.frame_base_vertices * self.scale) @ self.rotation.T
            + self.center
        )
        self.frame.vertices = o3d.utility.Vector3dVector(frame_vertices)
        self.frame.compute_vertex_normals()
        visualizer.update_geometry(self.mesh)
        visualizer.update_geometry(self.frame)


class PinchState:
    def __init__(self, side, enter_threshold, exit_threshold,
                 missing_timeout=0.18):
        self.side = side
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.missing_timeout = missing_timeout
        self.pinching = False
        self.controls_ball = False
        self.sequence = None
        self.point = None
        self.hand_frame = np.eye(3)
        self.last_seen = -np.inf
        self.distance = np.inf
        self.geometry = o3d.geometry.TriangleMesh.create_sphere(
            radius=5.0, resolution=12
        )
        self.geometry.compute_vertex_normals()
        self.geometry.paint_uniform_color(PINCH_COLORS[side])
        self.base_vertices = np.asarray(self.geometry.vertices).copy()
        self.added_to_viewer = False

    def update(self, world_landmarks, timestamp, ball, grab_tolerance,
               sequence, scene_point=None):
        self.last_seen = timestamp
        self.distance = np.linalg.norm(
            world_landmarks[4] - world_landmarks[8]
        )
        if scene_point is not None:
            self.point = scene_point
        self.hand_frame = palm_frame(world_landmarks)
        entered = False
        if (not self.pinching and scene_point is not None
                and self.distance <= self.enter_threshold):
            self.pinching = True
            self.sequence = sequence
            self.controls_ball = ball.contains(
                self.point, tolerance=grab_tolerance
            )
            entered = True
        elif self.pinching and self.distance >= self.exit_threshold:
            self.release()
        return entered

    def release(self):
        self.pinching = False
        self.controls_ball = False
        self.sequence = None

    def expire_if_missing(self, timestamp):
        if self.pinching and timestamp - self.last_seen > self.missing_timeout:
            self.release()

    def update_geometry(self, visualizer):
        if self.pinching and self.point is not None:
            self.geometry.vertices = o3d.utility.Vector3dVector(
                self.base_vertices + self.point
            )
            self.geometry.compute_vertex_normals()
            if not self.added_to_viewer:
                visualizer.add_geometry(self.geometry, reset_bounding_box=False)
                self.added_to_viewer = True
            visualizer.update_geometry(self.geometry)
        elif self.added_to_viewer:
            visualizer.remove_geometry(self.geometry, reset_bounding_box=False)
            self.added_to_viewer = False


class BallController:
    def __init__(self, ball, minimum_scale=0.35, maximum_scale=2.0,
                 scale_sensitivity=0.5):
        self.ball = ball
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        self.scale_sensitivity = scale_sensitivity
        self.signature = ()
        self.baseline = None

    def update(self, pinch_states):
        controllers = sorted(
            (state for state in pinch_states.values()
             if state.pinching and state.controls_ball),
            key=lambda state: state.sequence,
        )[:2]
        signature = tuple((state.side, state.sequence) for state in controllers)
        if signature != self.signature:
            self.signature = signature
            self._capture_baseline(controllers)
        if len(controllers) == 1:
            self._update_single(controllers[0])
        elif len(controllers) == 2:
            self._update_double(controllers[0], controllers[1])

    def _capture_baseline(self, controllers):
        if not controllers:
            self.baseline = None
            return
        self.baseline = {
            'center': self.ball.center.copy(),
            'rotation': self.ball.rotation.copy(),
            'scale': self.ball.scale,
            'first_point': controllers[0].point.copy(),
            'first_frame': controllers[0].hand_frame.copy(),
        }
        if len(controllers) == 2:
            vector = controllers[1].point - controllers[0].point
            self.baseline['two_vector'] = vector
            self.baseline['two_distance'] = max(np.linalg.norm(vector), 1e-6)

    def _update_single(self, controller):
        baseline = self.baseline
        delta_rotation = controller.hand_frame @ baseline['first_frame'].T
        self.ball.center = (
            baseline['center'] + controller.point - baseline['first_point']
        )
        self.ball.rotation = delta_rotation @ baseline['rotation']

    def _update_double(self, first, second):
        baseline = self.baseline
        vector = second.point - first.point
        distance = max(np.linalg.norm(vector), 1e-6)
        raw_ratio = distance / baseline['two_distance']
        scale_ratio = raw_ratio ** self.scale_sensitivity
        new_scale = np.clip(
            baseline['scale'] * scale_ratio,
            self.minimum_scale,
            self.maximum_scale,
        )
        applied_ratio = new_scale / baseline['scale']
        delta_rotation = rotation_between_vectors(
            baseline['two_vector'], vector
        )
        relative_center = baseline['center'] - baseline['first_point']
        self.ball.center = (
            first.point + delta_rotation @ (relative_center * applied_ratio)
        )
        self.ball.rotation = delta_rotation @ baseline['rotation']
        self.ball.scale = new_scale
