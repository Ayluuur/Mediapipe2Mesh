"""Bounded camera prefetch and persistent, per-hand IK processes."""

import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from mediapipe2mesh.solve_snapshot import SolveSnapshot


_converter = None
_generation = None


@dataclass
class DetectionPacket:
    frame: object
    timestamp: float
    result: object
    capture_seconds: float
    mediapipe_seconds: float

    def __iter__(self):
        return iter((self.frame, self.timestamp, self.result))


def _solve_hand(options, generation, landmarks):
    # Imported inside the child: no camera or Open3D objects cross processes.
    global _converter, _generation
    from mediapipe2mesh.ik import Keypoints2Mano

    if _converter is None:
        _converter = Keypoints2Mano(**options)
        # Close nested pools before this persistent hand process exits.
        from multiprocessing.util import Finalize
        Finalize(None, _converter.close, exitpriority=20)
        print('[IK {}] coordinator_pid={} jacobian_backend={} workers={}'.format(
            options.get('side'), os.getpid(), _converter.wrapper.jacobian_backend,
            _converter.wrapper.jacobian_workers,
        ), flush=True)
    if _generation != generation:
        _converter.reset()
        _generation = generation
    previous_pids = _converter.wrapper.worker_pids.copy()
    started = time.perf_counter()
    _converter.get_mano_params(landmarks)
    seconds = time.perf_counter() - started
    if _converter.wrapper.worker_pids != previous_pids:
        print('[IK {}] coordinator_pid={} executed_jacobian_pids={}'.format(
            options.get('side'), os.getpid(), sorted(_converter.wrapper.worker_pids)
        ), flush=True)
    return SolveSnapshot(
        _converter.get_camera_oriented_vertices().copy(),
        _converter.get_camera_oriented_keypoints().copy(),
        _converter.get_skeleton(),
        seconds,
    )


def _worker_report():
    return {
        'coordinator_pid': os.getpid(),
        'jacobian_pids': sorted(_converter.wrapper.worker_pids),
        'workers': _converter.wrapper.worker_stats,
    }


def print_parallel_config(config):
    from mediapipe2mesh.config import PROJECT_ROOT
    print('[parallel] Python={} project={} main_pid={} logical_cpus={} '
          'hand_backend={} jacobian_backend={} workers_per_hand={}'.format(
              sys.executable, PROJECT_ROOT, os.getpid(), os.cpu_count(),
              getattr(config.mano, 'executor', 'process'),
              getattr(config.mano, 'jacobian_backend', 'thread'),
              getattr(config.mano, 'jacobian_workers', 1),
          ), flush=True)


class HandExecutor:
    """One process per side preserves warm starts and avoids the Python GIL."""

    def __init__(self, backend='process'):
        if backend not in ('process', 'thread'):
            raise ValueError('mano.executor must be process or thread')
        self.backend = backend
        self.pools = {}
        self.converters = {}

    def __enter__(self):
        return self

    def submit_hand(self, state, landmarks):
        if state.side not in self.pools:
            if self.backend == 'process':
                pool = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context('spawn')
                )
            else:
                pool = ThreadPoolExecutor(max_workers=1)
            self.pools[state.side] = pool
        pool = self.pools[state.side]
        if self.backend == 'thread':
            self.converters[state.side] = state.converter
            return pool.submit(state._solve, landmarks)
        return pool.submit(
            _solve_hand, state.solver_options, state.track_generation, landmarks
        )

    def __exit__(self, *args):
        for pool in self.pools.values():
            pool.shutdown(wait=True)
        for converter in self.converters.values():
            converter.close()

    def worker_reports(self):
        """Call between frames for a diagnostic snapshot, never in rendering."""
        if self.backend == 'process':
            jobs = {side: pool.submit(_worker_report) for side, pool in self.pools.items()}
            return {side: job.result() for side, job in jobs.items()}
        return {side: {'coordinator_pid': os.getpid(),
                       'jacobian_pids': sorted(converter.wrapper.worker_pids)}
                for side, converter in self.converters.items()}


class DetectionPipeline:
    """Prefetch at most one frame; a single worker owns detector.process().

    Frames, capture timestamps and landmarks always travel together. Detection
    remains chronological, preserving MediaPipe's temporal tracking state.
    """

    def __init__(self, capture, detector, mirror):
        self.capture = capture
        self.detector = detector
        self.mirror = mirror
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def _read(self):
        import cv2

        capture_started = time.perf_counter()
        ok, frame = self.capture.read()
        timestamp = time.perf_counter()
        capture_seconds = timestamp - capture_started
        if not ok:
            return None
        mediapipe_started = time.perf_counter()
        if self.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.detector.process(rgb)
        return DetectionPacket(
            frame, timestamp, result, capture_seconds,
            time.perf_counter() - mediapipe_started,
        )

    def read(self):
        if self.future is None:
            self.future = self.pool.submit(self._read)
        packet = self.future.result()
        if packet is not None:
            self.future = self.pool.submit(self._read)
        return packet

    def close(self):
        # Finish any camera read/process before their owners release resources.
        self.pool.shutdown(wait=True)
