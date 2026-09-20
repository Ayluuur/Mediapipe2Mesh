"""Run with python -m mediapipe2mesh.apps.benchmark; no camera required."""

import time
import argparse
from itertools import product
from types import SimpleNamespace

from mediapipe2mesh.apps.parallel import HandExecutor


def main():
    import numpy as np
    from mediapipe2mesh.config import PROJECT_ROOT
    from mediapipe2mesh.ik import Keypoints2Mano

    mapping = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5,
               6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 1, 2, 4])
    parser.add_argument('--jacobian-backend', choices=['thread', 'process'],
                        default='process')
    parser.add_argument('--backends', nargs='+', choices=['thread', 'process'],
                        default=['process'])
    args = parser.parse_args()
    if args.frames < 1 or any(not 0 <= w <= 45 for w in args.workers):
        parser.error('frames must be positive; workers must be in [0, 45]')
    frames = args.frames
    for backend, workers in product(args.backends, args.workers):
        states = []
        inputs = []
        for side in ('left', 'right'):
            options = dict(
                model_path=str(PROJECT_ROOT / ('MANO_' + side.upper() + '.npz')),
                side=side, max_iter=5, pose_smoothing=.7, mirrored_input=False,
                jacobian_workers=workers,
                jacobian_backend=args.jacobian_backend,
            )
            converter = Keypoints2Mano(**options)

            def solve(landmarks, hand=converter):
                hand.get_mano_params(landmarks)
                return (hand.get_camera_oriented_vertices().copy(),
                        hand.get_camera_oriented_keypoints().copy(),
                        hand.get_skeleton())

            states.append(SimpleNamespace(
                side=side, solver_options=options, track_generation=0, _solve=solve,
                converter=converter,
            ))
            sequence = []
            for index in range(frames):
                pose = np.zeros((16, 3))
                pose[1:, 2] = .5 + .35 * np.sin(index * .15)
                converter.mesh.set_params(pose_abs=pose)
                sequence.append(converter.mesh.keypoints[mapping].copy() / 1000)
            inputs.append(sequence)
        with HandExecutor(backend) as executor:
            # Exclude lazy worker startup/model loading from steady-state time.
            jobs = [executor.submit_hand(state, sequence[0])
                    for state, sequence in zip(states, inputs)]
            for job in jobs:
                job.result()
            started = time.perf_counter()
            for index in range(frames):
                jobs = [executor.submit_hand(state, sequence[index])
                        for state, sequence in zip(states, inputs)]
                for job in jobs:
                    job.result()
            elapsed = time.perf_counter() - started
            print('{} jacobian={} workers={}: {:.1f} two-hand frames/s, {:.2f} ms/frame'.format(
                backend, args.jacobian_backend, workers, frames / elapsed, elapsed * 1000 / frames
            ), flush=True)
            print('Actual workers: {}'.format(executor.worker_reports()), flush=True)
        for state in states:
            state.converter.close()


if __name__ == '__main__':
    main()
