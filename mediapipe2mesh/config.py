"""JSON configuration loading with recursive default overrides."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / 'configs'


class ConfigNode:
    """Dictionary wrapper providing read-only attribute-style access."""

    def __init__(self, values):
        self._values = values

    def __getattr__(self, name):
        try:
            value = self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return ConfigNode(value) if isinstance(value, dict) else value

    def as_dict(self):
        return dict(self._values)


def _merge(base, override):
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = value
    return output


def load_config(default_name, custom_path=None):
    default_path = CONFIG_ROOT / default_name
    with default_path.open('r', encoding='utf-8') as stream:
        values = json.load(stream)
    if custom_path is not None:
        path = Path(custom_path).expanduser().resolve()
        with path.open('r', encoding='utf-8') as stream:
            values = _merge(values, json.load(stream))
    return ConfigNode(values)


def config_path_from_argv():
    """Accept zero arguments or one custom JSON path without argparse."""
    if len(sys.argv) > 2:
        raise SystemExit('Usage: {} [config.json]'.format(sys.argv[0]))
    return sys.argv[1] if len(sys.argv) == 2 else None

