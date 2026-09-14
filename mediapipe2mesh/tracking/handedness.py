"""Stable handedness mapping and wrist-based track association."""

import numpy as np


def opposite_side(side):
    return 'right' if side == 'left' else 'left'


def map_handedness(label, mapping='direct', mirrored_input=True):
    side = label.lower()
    if mapping == 'auto':
        return side if mirrored_input else opposite_side(side)
    if mapping == 'swapped':
        return opposite_side(side)
    if mapping == 'direct':
        return side
    raise ValueError("handedness_map must be 'auto', 'direct', or 'swapped'")


class HandednessResolver:
    """Associate wrists over time and debounce handedness labels."""

    def __init__(self, confirm_frames=5, track_timeout=0.35,
                 max_wrist_distance=0.30):
        self.confirm_frames = confirm_frames
        self.track_timeout = track_timeout
        self.max_wrist_distance = max_wrist_distance
        self.pending = {}

    def resolve(self, raw_detections, states, timestamp):
        resolved = {}
        used_detections = set()
        used_states = set()
        for state in states.values():
            if state.screen_wrist is not None and timestamp - state.last_seen >= self.track_timeout:
                state.deactivate()
        recent_states = [
            state for state in states.values()
            if (state.screen_wrist is not None and
                timestamp - state.last_seen < self.track_timeout)
        ]
        pairs = []
        for index, detection in enumerate(raw_detections):
            for state in recent_states:
                distance = np.linalg.norm(
                    detection['wrist'] - state.screen_wrist
                )
                pairs.append((distance, index, state.side))
        for distance, index, state_side in sorted(pairs):
            if distance > self.max_wrist_distance:
                break
            if index in used_detections or state_side in used_states:
                continue
            detection = raw_detections[index]
            stable_side = state_side
            # Identity belongs to the active spatial track, not the latest
            # handedness label. Reclassify only after disappearance/re-entry.
            detection['label_was_stabilized'] = (
                detection['raw_side'] != stable_side
            )
            resolved[stable_side] = detection
            used_detections.add(index)
            used_states.add(state_side)
        candidates = {}
        active_sides = {state.side for state in recent_states}
        for index, detection in enumerate(raw_detections):
            if index in used_detections:
                continue
            side = detection['raw_side']
            if side in resolved or side in active_sides:
                continue  # Never overwrite a spatially matched track.
            # With only one detection, an unmatched live hand may have jumped.
            # Wait for its timeout instead of creating the opposite MANO hand.
            if len(raw_detections) == 1 and recent_states:
                continue
            previous = candidates.get(side)
            if previous is None or detection['score'] > previous['score']:
                candidates[side] = detection
        pending = {}
        for side, detection in candidates.items():
            previous = self.pending.get(side)
            continuous = (previous is not None
                          and timestamp - previous[2] < self.track_timeout
                          and np.linalg.norm(detection['wrist'] - previous[0])
                          <= self.max_wrist_distance)
            count = previous[1] + 1 if continuous else 1
            if count >= self.confirm_frames:
                detection['label_was_stabilized'] = False
                resolved[side] = detection
            else:
                pending[side] = (detection['wrist'].copy(), count, timestamp)
        self.pending = pending  # A missing candidate breaks consecutive confirmation.
        return resolved
