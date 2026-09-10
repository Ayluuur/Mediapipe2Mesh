"""Backward-compatible imports for the reorganized IK package."""

from mediapipe2mesh.ik.keypoints_to_mano import (
    FINGER_CHAINS,
    MANO_KEYPOINT_PARENTS,
    N_POSE,
    Keypoints2Mano,
    PoseOnlyWrapper,
)

__all__ = [
    'FINGER_CHAINS', 'MANO_KEYPOINT_PARENTS', 'N_POSE',
    'Keypoints2Mano', 'PoseOnlyWrapper',
]
