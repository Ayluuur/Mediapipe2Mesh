"""Detached bone-transform snapshots; column vectors, millimeters, radians."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class SkeletonPose:
    """16 MANO bones (tips are surface landmarks, not articulated joints).

    ``transforms`` map joint-local column vectors into ``space``.
    ``local_transforms`` map into the parent frame; the root maps into ``space``.
    Quaternions use xyzw order. Every snapshot owns its arrays.
    """

    side: str
    space: str
    names: tuple
    parents: np.ndarray
    transforms: np.ndarray
    rest_positions: np.ndarray
    fit_error_mm: float
    mirrored_local_axes: bool = False

    @property
    def positions(self):
        return self.transforms[:, :3, 3].copy()

    @property
    def rotations(self):
        return self.transforms[:, :3, :3].copy()

    @property
    def quaternions(self):
        return Rotation.from_matrix(self.rotations).as_quat()

    @property
    def displacements(self):
        """Position offsets from bind pose in the same space (mm)."""
        return self.positions - self.rest_positions

    @property
    def local_transforms(self):
        result = self.transforms.copy()
        for joint in range(1, len(self.parents)):
            result[joint] = (np.linalg.inv(self.transforms[self.parents[joint]])
                             @ self.transforms[joint])
        return result

    @property
    def local_positions(self):
        return self.local_transforms[:, :3, 3]

    @property
    def local_rotations(self):
        return self.local_transforms[:, :3, :3]

    @property
    def local_quaternions(self):
        return Rotation.from_matrix(self.local_rotations).as_quat()

    @property
    def local_axis_angles(self):
        return Rotation.from_matrix(self.local_rotations).as_rotvec()

    def to_scene(self, translation):
        """Convert a camera snapshot to Open3D with the displayed wrist in mm."""
        if self.space != 'camera':
            raise ValueError('to_scene requires a camera-space snapshot')
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError('translation must be a finite (3,) vector in mm')
        basis = np.diag([1.0, -1.0, -1.0])
        transform = np.eye(4)
        transform[:3, :3] = basis
        transform[:3, 3] = translation
        return SkeletonPose(
            self.side, 'scene', self.names, self.parents.copy(),
            transform @ self.transforms, self.rest_positions @ basis.T,
            self.fit_error_mm,
            self.mirrored_local_axes,
        )

    def to_dict(self):
        """JSON-compatible public data without NumPy objects."""
        data = dict(side=self.side, space=self.space, units='mm',
                    quaternion_order='xyzw', rotation_convention='column_vectors',
                    mirrored_local_axes=self.mirrored_local_axes,
                    names=list(self.names), fit_error_mm=self.fit_error_mm)
        for name in ('parents', 'positions', 'rest_positions', 'rotations', 'quaternions',
                     'displacements', 'local_positions', 'local_rotations',
                     'local_quaternions', 'local_axis_angles', 'transforms',
                     'local_transforms'):
            data[name] = getattr(self, name).tolist()
        return data
