import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mediapipe2mesh.apps.launcher import main, mode_config


class LauncherTests(unittest.TestCase):
    def test_defaults_and_independent_overrides(self):
        preview = mode_config('interaction')
        parallel = mode_config('multicore')
        self.assertEqual(preview.mano.jacobian_workers, 1)
        self.assertEqual(parallel.mano.jacobian_workers, 8)
        self.assertEqual(parallel.mano.jacobian_backend, 'process')
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'custom.json'
            file.write_text(json.dumps({'mano': {'jacobian_workers': 4}}))
            self.assertEqual(mode_config('multicore', file).mano.jacobian_workers, 4)
            self.assertEqual(mode_config('interaction').mano.jacobian_workers, 1)
        self.assertEqual(preview.button.center, parallel.button.center)

    def test_cli_mode_and_menu_routing(self):
        with patch('mediapipe2mesh.apps.launcher.show_launcher') as menu:
            main([])
            menu.assert_called_once()
        with patch('mediapipe2mesh.apps.launcher.run_mode') as run:
            main(['--mode', 'multicore'])
            run.assert_called_once_with('multicore', None)
        with patch('mediapipe2mesh.apps.launcher.run_mode') as run:
            main(['--mode', 'viewer'])
            run.assert_called_once_with('viewer', None)


if __name__ == '__main__':
    unittest.main()
