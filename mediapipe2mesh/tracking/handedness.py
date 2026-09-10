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

    def __init__(self, confirm_frames=5, track_timeout=0.5,
                 max_wrist_distance=0.30):
        self.confirm_frames = confirm_frames
        self.track_timeout = track_timeout
        self.max_wrist_distance = max_wrist_distance

    def resolve(self, raw_detections, states, timestamp):
        resolved = {}
        used_detections = set()
        used_states = set()
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
            state = states[state_side]
            stable_side = state_side
            if detection['raw_side'] == state_side:
                state.side_candidate = None
                state.side_candidate_frames = 0
            else:
                if state.side_candidate == detection['raw_side']:
                    state.side_candidate_frames += 1
                else:
                    state.side_candidate = detection['raw_side']
                    state.side_candidate_frames = 1
                candidate = states[detection['raw_side']]
                free = (timestamp - candidate.last_seen >= self.track_timeout
                        and detection['raw_side'] not in resolved)
                if state.side_candidate_frames >= self.confirm_frames and free:
                    stable_side = detection['raw_side']
                    state.side_candidate = None
                    state.side_candidate_frames = 0
            detection['label_was_stabilized'] = (
                detection['raw_side'] != stable_side
            )
            resolved[stable_side] = detection
            used_detections.add(index)
            used_states.add(state_side)
        for index, detection in enumerate(raw_detections):
            if index in used_detections:
                continue
            side = detection['raw_side']
            detection['label_was_stabilized'] = False
            previous = resolved.get(side)
            if previous is None or detection['score'] > previous['score']:
                resolved[side] = detection
        return resolved

