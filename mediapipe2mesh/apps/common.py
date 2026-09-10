"""Shared camera, MediaPipe, and hand-state application helpers."""

import cv2
import numpy as np

from mediapipe2mesh.tracking import map_handedness


SIDES = ('left', 'right')


def open_camera(config):
    capture = cv2.VideoCapture(config.index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError('Cannot open camera {}'.format(config.index))
    return capture


def create_hand_detector(mp_hands, config):
    return mp_hands.Hands(
        static_image_mode=False,
        model_complexity=config.model_complexity,
        max_num_hands=config.max_hands,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )


def extract_detections(results, handedness_map, mirrored_input):
    output = []
    for screen, world, hand_info in zip(
            results.multi_hand_landmarks or [],
            results.multi_hand_world_landmarks or [],
            results.multi_handedness or []):
        classification = hand_info.classification[0]
        wrist = screen.landmark[0]
        output.append({
            'raw_side': map_handedness(
                classification.label, handedness_map, mirrored_input
            ),
            'screen': screen,
            'world': world,
            'score': classification.score,
            'wrist': np.array([wrist.x, wrist.y]),
        })
    return output


def update_hand_track(state, detection, timestamp):
    if timestamp - state.last_seen > 0.35:
        state.begin_track()
    landmarks = np.array([
        [point.x, point.y, point.z]
        for point in detection['world'].landmark
    ], dtype=np.float64)
    filtered = state.filter(landmarks, timestamp)
    state.pending_landmarks = filtered
    state.last_seen = timestamp
    state.screen_wrist = detection['wrist']
    state.display_wrist = state.position_filter(
        detection['wrist'], timestamp
    )
    return filtered


def retire_missing_single_hand(raw_detections, detections, states,
                               on_retire=None):
    if len(raw_detections) != 1 or len(detections) != 1:
        return
    only_side = next(iter(detections))
    for side, state in states.items():
        if side != only_side:
            state.deactivate()
            if on_retire is not None:
                on_retire(side)

