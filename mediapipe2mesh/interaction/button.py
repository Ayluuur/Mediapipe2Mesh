"""Depth button and fingertip contact in the existing scene coordinates."""

import numpy as np
import open3d as o3d

from mediapipe2mesh.visualization import SceneMapper


def mano_index_tip_to_scene(hand_state, mapper=None, depth=0.0):
    if hand_state.keypoints is None or hand_state.display_wrist is None:
        return None
    points = (mapper or SceneMapper()).hand_keypoints(
        hand_state.keypoints, hand_state.display_wrist, depth,
        translation=hand_state.scene_translation,
    )
    return points[16].copy()  # MANO I3, equivalent to MediaPipe landmark 8.


class DepthButton:
    """Front face points toward -Z; pressing moves the cap along +Z.

    A fingertip must contact the front face or cross it from the viewer side.
    Each hand keeps its own contact state; losing detection releases that hand.
    """

    def __init__(self, center, width, height, travel, tip_radius):
        self.center = np.asarray(center, dtype=float)
        self.width = float(width)
        self.height = float(height)
        self.travel = float(travel)
        self.tip_radius = float(tip_radius)
        self.contacts = set()
        self.previous = {}
        self.pressed = False
        self.press_count = 0
        self.cap = self._box(width, height, 10.0)
        self.cap_vertices = np.asarray(self.cap.vertices).copy()
        self.base = self._box(width + 12.0, height + 12.0, 8.0)
        self.base.translate(self.center + [0, 0, travel + 10.0])
        self.base.paint_uniform_color((0.22, 0.25, 0.30))
        self._update_cap()

    @staticmethod
    def _box(width, height, thickness):
        mesh = o3d.geometry.TriangleMesh.create_box(width, height, thickness)
        mesh.translate([-width / 2, -height / 2, 0])
        mesh.compute_vertex_normals()
        return mesh

    def update(self, points):
        points = {side: np.asarray(point, dtype=float).copy()
                  for side, point in points.items()
                  if point is not None and np.all(np.isfinite(point))}
        contacts = set()
        for side, point in points.items():
            local = point - self.center
            margin = self.tip_radius + (3.0 if side in self.contacts else 0.0)
            inside_xy = (abs(local[0]) <= self.width / 2 + margin
                         and abs(local[1]) <= self.height / 2 + margin)
            if side in self.contacts:
                hit = inside_xy and -margin <= local[2] <= self.travel + 10 + margin
            else:
                hit = inside_xy and abs(local[2]) <= self.tip_radius
                previous = self.previous.get(side)
                if previous is not None:
                    start = previous - self.center
                    front = -self.tip_radius
                    if start[2] < front <= local[2]:
                        fraction = (front - start[2]) / (local[2] - start[2])
                        crossing = start + fraction * (local - start)
                        hit |= (abs(crossing[0]) <= self.width / 2 + self.tip_radius
                                and abs(crossing[1]) <= self.height / 2 + self.tip_radius)
            if hit:
                contacts.add(side)
        pressed = bool(contacts)
        if pressed and not self.pressed:
            self.press_count += 1
        self.pressed = pressed
        self.contacts = contacts
        self.previous = points
        self._update_cap()

    def _update_cap(self):
        offset = self.center + [0, 0, self.travel if self.pressed else 0.0]
        self.cap.vertices = o3d.utility.Vector3dVector(self.cap_vertices + offset)
        self.cap.paint_uniform_color(
            (0.15, 1.0, 0.35) if self.pressed else (0.25, 0.48, 0.85)
        )

    def add_to(self, visualizer):
        visualizer.add_geometry(self.base, reset_bounding_box=False)
        visualizer.add_geometry(self.cap, reset_bounding_box=False)

    def update_geometry(self, visualizer):
        visualizer.update_geometry(self.cap)
