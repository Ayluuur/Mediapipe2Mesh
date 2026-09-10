"""Compatibility entry point for the pinch interaction demo."""

from mediapipe2mesh.apps.interaction_app import run_interaction
from mediapipe2mesh.config import config_path_from_argv, load_config


if __name__ == '__main__':
    run_interaction(load_config('interaction.json', config_path_from_argv()))
