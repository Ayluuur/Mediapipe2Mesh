"""Compatibility entry point for the MANO webcam viewer."""

from mediapipe2mesh.apps.viewer_app import run_viewer
from mediapipe2mesh.config import config_path_from_argv, load_config


if __name__ == '__main__':
    run_viewer(load_config('viewer.json', config_path_from_argv()))
