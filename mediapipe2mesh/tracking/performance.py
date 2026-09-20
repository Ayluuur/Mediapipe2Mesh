"""Separate solver computation from the rate of newly displayed meshes."""

from collections import deque

from mediapipe2mesh.solve_snapshot import SolveSnapshot


class FrameRates:
    def __init__(self, timestamp):
        self.started = timestamp
        self.frames = 0
        self.capture_fps = 0.0
        self.versions = dict(left=0, right=0)
        self.counts = dict(left=0, right=0)
        self.mano_fps = dict(left=0.0, right=0.0)
        self.phase_seconds = {
            name: deque(maxlen=60)
            for name in ('capture', 'mediapipe', 'main_render')
        }

    def record_phases(self, capture, mediapipe, main_render):
        for name, value in (
                ('capture', capture), ('mediapipe', mediapipe),
                ('main_render', main_render)):
            if value >= 0:
                self.phase_seconds[name].append(value)

    def mean_phase_ms(self, name):
        samples = self.phase_seconds[name]
        return 1000.0 * sum(samples) / len(samples) if samples else 0.0

    def update(self, states, visible, timestamp):
        self.frames += 1
        for side in visible:
            version = states[side].result_version
            if version != self.versions[side]:
                self.counts[side] += 1
                self.versions[side] = version
        elapsed = timestamp - self.started
        if elapsed >= .5:
            self.capture_fps = self.frames / elapsed
            self.mano_fps = {side: count / elapsed for side, count in self.counts.items()}
            self.counts = dict(left=0, right=0)
            self.frames = 0
            self.started = timestamp

    def labels(self, states, timestamp, timeout=.35):
        def ik(side):
            state = states[side]
            seconds = state.mean_ik_seconds
            if (timestamp - state.last_seen >= timeout
                    or state.vertices is None or seconds <= 0):
                return '--'
            return '{:.1f} ({:.1f}ms)'.format(1.0 / seconds, seconds * 1000)
        return (
            'IK compute FPS L:{} R:{}'.format(ik('left'), ik('right')),
            'MANO FPS L:{:.1f} R:{:.1f} | Capture/MP:{:.1f}'.format(
                self.mano_fps['left'], self.mano_fps['right'], self.capture_fps),
            'Time ms Capture:{:.1f} MediaPipe:{:.1f} Main/render:{:.1f}'.format(
                self.mean_phase_ms('capture'), self.mean_phase_ms('mediapipe'),
                self.mean_phase_ms('main_render')),
        )

    def draw(self, frame, states, timestamp, timeout=.35):
        import cv2
        for line, label in enumerate(self.labels(states, timestamp, timeout)):
            width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .58, 2)[0][0]
            scale = min(.58, .58 * max(1, frame.shape[1] - 24) / max(1, width))
            cv2.putText(frame, label, (12, 28 + 24 * line),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (80, 255, 255) if line == 0 else (80, 255, 80), 1, cv2.LINE_AA)
