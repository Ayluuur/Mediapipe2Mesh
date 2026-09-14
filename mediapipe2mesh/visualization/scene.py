"""Open3D scene setup and MediaPipe-to-scene coordinate mapping."""

import numpy as np
import open3d as o3d


SCENE_WIDTH_MM = 300.0
SCENE_HEIGHT_MM = 240.0
CAMERA_TO_OPEN3D = np.array([1.0, -1.0, -1.0])
CAMERA_NEAR_MM = 50.0
CAMERA_FAR_MM = 1500.0
CAMERA_FOV_DEGREES = 60.0


def scroll_zoom(visualizer, x_offset, y_offset):
    """Positive wheel motion brings the camera closer (zooms in)."""
    visualizer.get_view_control().scale(-y_offset)
    return False  # Only the camera changed; geometry needs no update.


def create_visualizer(window_name, on_calibrate=None):
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name=window_name)
    visualizer.register_mouse_scroll_callback(scroll_zoom)
    if on_calibrate is not None:
        def calibrate(visualizer):
            on_calibrate()
            return False
        for key in (257, 335):  # GLFW Enter and keypad Enter.
            visualizer.register_key_callback(key, calibrate)
    return visualizer


class SceneMapper:
    @staticmethod
    def palm_scale(screen_landmarks, world_landmarks, reference_keypoints, width, height,
                   segment_corrections=None):
        """Match same-frame projections, using fixed MANO bone lengths.

        Rotation shortens both the observed and predicted projected segment.
        Reject near-edge-on segments instead of dividing by a vanishing length.
        """
        if hasattr(screen_landmarks, 'landmark'):
            screen = np.array([[p.x, p.y] for p in screen_landmarks.landmark])
        else:
            screen = np.asarray(screen_landmarks, dtype=float)[:, :2]
        if hasattr(world_landmarks, 'landmark'):
            world = np.array([[p.x, p.y, p.z] for p in world_landmarks.landmark])
        else:
            world = np.asarray(world_landmarks, dtype=float)
        pixels = screen * [width, height]
        screen_pairs = ((0, 9), (5, 17), (5, 9), (9, 13), (13, 17))
        mano_pairs = ((0, 4), (1, 7), (1, 4), (4, 10), (10, 7))
        ratios, weights, indices = [], [], []
        for index, ((a, b), (c, d)) in enumerate(zip(screen_pairs, mano_pairs)):
            segment = world[a] - world[b]
            length = np.linalg.norm(segment)
            if not np.isfinite(length) or length < 1e-8:
                continue
            projection = np.linalg.norm(segment[:2]) / length
            observed = np.linalg.norm(pixels[a] - pixels[b])
            fixed_length = np.linalg.norm(reference_keypoints[c] - reference_keypoints[d])
            if (projection < 0.3 or not np.isfinite(observed) or observed < 5
                    or not np.isfinite(fixed_length) or fixed_length < 1):
                continue
            ratios.append(fixed_length * projection / observed)
            weights.append(observed ** 2)
            indices.append(index)
        normal = np.cross(world[5] - world[0], world[17] - world[0])
        normal_length = np.linalg.norm(normal)
        facing = (abs(normal[2]) / normal_length
                  if np.isfinite(normal_length) and normal_length > 1e-10 else 0.0)
        if not ratios:
            return None
        if len(ratios) < 2:
            # At 90 degrees only the long wrist-to-MCP segment may remain.
            # Its learned scale remains useful for real approach/recede motion.
            if (segment_corrections is None or indices[0] not in segment_corrections
                    or weights[0] < 15.0 ** 2):
                return None
        raw_ratios = np.asarray(ratios)
        if segment_corrections is not None and facing < 0.8:
            # MANO and a person's palm need not have identical length/width
            # proportions. Normalize segment ratios to the same frontal scale
            # so switching to the long segment cannot widen XY.
            ratios = raw_ratios * np.array([
                segment_corrections.get(i, 1.0)
                for i in indices
            ])
        # Long, well-resolved segments dominate; tiny MCP gaps do not.
        order = np.argsort(ratios)
        cumulative = np.cumsum(np.asarray(weights)[order])
        index = np.searchsorted(cumulative, cumulative[-1] * 0.5)
        scale = float(np.asarray(ratios)[order[index]])
        if segment_corrections is not None and facing >= 0.8:
            for i, ratio in zip(indices, raw_ratios):
                correction = scale / ratio
                previous = segment_corrections.get(i, correction)
                segment_corrections[i] = previous + 0.1 * (correction - previous)
        return scale

    def screen_translation(self, screen_landmarks, keypoints, width, height, depth,
                           mm_per_pixel=None):
        """Recover wrist XY in MANO millimeters from observed palm magnification."""
        if hasattr(screen_landmarks, 'landmark'):
            screen = np.array([[p.x, p.y] for p in screen_landmarks.landmark])
        else:
            screen = np.asarray(screen_landmarks, dtype=float)[:, :2]
        pixels = screen * np.array([width, height])
        if mm_per_pixel is not None:
            if not np.all(np.isfinite(pixels[0])):
                return None
            wrist_xy = (pixels[0] - np.array([width, height]) * 0.5) * mm_per_pixel
            return np.array([wrist_xy[0], -wrist_xy[1], float(depth)])
        # Corresponding palm segments in MediaPipe and MANO joint order.
        screen_pairs = ((0, 9), (5, 17), (5, 9), (9, 13), (13, 17))
        mano_pairs = ((0, 4), (1, 7), (1, 4), (4, 10), (10, 7))
        ratios = []
        for (a, b), (c, d) in zip(screen_pairs, mano_pairs):
            pixel_length = np.linalg.norm(pixels[a] - pixels[b])
            model_length = np.linalg.norm(keypoints[c, :2] - keypoints[d, :2])
            if (np.isfinite(pixel_length) and np.isfinite(model_length)
                    and pixel_length >= 5.0 and model_length >= 1.0):
                ratios.append(model_length / pixel_length)
        if len(ratios) < 2 or not np.all(np.isfinite(pixels[0])):
            return None
        mm_per_pixel = float(np.median(ratios))
        wrist_xy = (pixels[0] - np.array([width, height]) * 0.5) * mm_per_pixel
        return np.array([wrist_xy[0], -wrist_xy[1], float(depth)])

    def wrist_translation(self, wrist, depth=0.0):
        translation = np.array([
            (wrist[0] - 0.5) * SCENE_WIDTH_MM,
            (0.5 - wrist[1]) * SCENE_HEIGHT_MM,
            float(depth),
        ])
        return translation

    def hand_vertices(self, vertices, wrist, depth=0.0, translation=None):
        return (vertices * CAMERA_TO_OPEN3D
                + (self.wrist_translation(wrist, depth) if translation is None
                   else translation))

    def hand_keypoints(self, keypoints, wrist, depth=0.0, translation=None):
        return (keypoints * CAMERA_TO_OPEN3D
                + (self.wrist_translation(wrist, depth) if translation is None
                   else translation))


def set_geometry_visible(visualizer, state, visible,
                         reset_bounding_box=False):
    if visible and not state.added_to_viewer:
        visualizer.add_geometry(
            state.mesh, reset_bounding_box=reset_bounding_box
        )
        state.added_to_viewer = True
    elif not visible and state.added_to_viewer:
        visualizer.remove_geometry(state.mesh, reset_bounding_box=False)
        state.added_to_viewer = False


def configure_view(visualizer, mode):
    control = visualizer.get_view_control()
    if mode == 'front':
        front = np.array([0.0, 0.0, -1.0])
    elif mode == 'depth':
        front = np.array([0.28, -0.16, -1.0])
        front /= np.linalg.norm(front)
    else:
        raise ValueError("view mode must be 'front' or 'depth'")
    control.set_front(front)
    control.set_lookat(np.zeros(3))
    control.set_up(np.array([0.0, 1.0, 0.0]))
    control.set_zoom(0.72)
    control.change_field_of_view(
        CAMERA_FOV_DEGREES - control.get_field_of_view()
    )
    control.set_constant_z_near(CAMERA_NEAR_MM)
    control.set_constant_z_far(CAMERA_FAR_MM)


def create_scene_bounds(visualizer, z_min=-80.0, z_max=80.0):
    options = visualizer.get_render_option()
    background = np.array([0.03, 0.03, 0.03])
    options.background_color = background
    options.light_on = True
    options.mesh_show_back_face = True
    bounds = o3d.geometry.PointCloud()
    bounds.points = o3d.utility.Vector3dVector(np.array([
        [-230.0, -180.0, z_min], [230.0, 180.0, z_max]
    ]))
    bounds.colors = o3d.utility.Vector3dVector(np.stack((background, background)))
    visualizer.add_geometry(bounds, reset_bounding_box=True)
    return bounds
