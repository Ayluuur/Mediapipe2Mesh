"""Open3D scene setup and MediaPipe-to-scene coordinate mapping."""

import numpy as np
import open3d as o3d


SCENE_WIDTH_MM = 300.0
SCENE_HEIGHT_MM = 240.0
CAMERA_TO_OPEN3D = np.array([1.0, -1.0, -1.0])
CAMERA_NEAR_MM = 50.0
CAMERA_FAR_MM = 1500.0
CAMERA_FOV_DEGREES = 60.0


class SceneMapper:
    def wrist_translation(self, wrist, depth=0.0):
        return np.array([
            (wrist[0] - 0.5) * SCENE_WIDTH_MM,
            (0.5 - wrist[1]) * SCENE_HEIGHT_MM,
            float(depth),
        ])

    def hand_vertices(self, vertices, wrist, depth=0.0):
        return (vertices * CAMERA_TO_OPEN3D
                + self.wrist_translation(wrist, depth))

    def hand_keypoints(self, keypoints, wrist, depth=0.0):
        return (keypoints * CAMERA_TO_OPEN3D
                + self.wrist_translation(wrist, depth))


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
