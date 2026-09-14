"""Webcam MANO viewer application."""

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d

from mediapipe2mesh.apps.common import (
    SIDES,
    create_hand_detector,
    extract_detections,
    open_camera,
    retire_missing_single_hand,
    update_hand_track,
    update_hand_scene_position,
)
from mediapipe2mesh.tracking import (
    HandState,
    HandednessResolver,
    MultiHandDepthEstimator,
)
from mediapipe2mesh.visualization import (
    SceneMapper,
    configure_view,
    create_scene_bounds,
    create_visualizer,
    set_geometry_visible,
)


def validate_config(config):
    if not 0.0 <= config.mano.pose_smoothing <= 1.0:
        raise ValueError('mano.pose_smoothing must be in [0, 1]')
    if config.tracking.handedness_confirm_frames < 1:
        raise ValueError('tracking.handedness_confirm_frames must be positive')
    if not (0.0 <= config.depth_estimation.minimum_depth
            < config.depth_estimation.maximum_depth):
        raise ValueError('Invalid depth_estimation range')
    reference = config.depth_estimation.reference_depth
    if not (config.depth_estimation.minimum_depth <= reference
            <= config.depth_estimation.maximum_depth):
        raise ValueError('depth_estimation.reference_depth is outside range')
    if config.depth_estimation.calibration_frames < 1:
        raise ValueError('depth_estimation.calibration_frames must be positive')
    if config.depth_estimation.motion_gain <= 0.0:
        raise ValueError('depth_estimation.motion_gain must be positive')


def run_viewer(config):
    validate_config(config)
    states = {
        side: HandState(
            side, config.mano.iterations, config.mano.pose_smoothing,
            getattr(config.tracking, 'position_filter', None),
        ) for side in SIDES
    }
    resolver = HandednessResolver(
        confirm_frames=config.tracking.handedness_confirm_frames
    )
    depth_tracker = MultiHandDepthEstimator(config.depth_estimation, SIDES)
    depth_estimators = depth_tracker.estimators
    depths = {side: 0.0 for side in SIDES}

    def request_depth_calibration():
        if config.depth_estimation.enabled:
            depth_tracker.request_calibration(time.perf_counter())

    visualizer = create_visualizer(
        config.viewer.window_name, on_calibrate=request_depth_calibration
    )
    scene_bounds = create_scene_bounds(visualizer, -60.0, 60.0)
    view_mode = config.viewer.initial_view
    configure_view(visualizer, view_mode)
    mapper = SceneMapper()

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    hands = create_hand_detector(mp_hands, config.detector)
    capture = open_camera(config.camera)
    fps = 0.0
    fps_frames = 0
    fps_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                if config.camera.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True
                for state in states.values():
                    state.collect_result()
                raw = extract_detections(
                    results,
                    config.tracking.handedness_map,
                    config.camera.mirror,
                )
                detections = resolver.resolve(raw, states, now)
                if config.depth_estimation.enabled:
                    scene_scales = {
                        side: mapper.palm_scale(
                            detection['screen'], detection['world'],
                            states[side].scene_reference_keypoints,
                            frame.shape[1], frame.shape[0],
                            segment_corrections=states[side].scene_segment_corrections,
                        ) for side, detection in detections.items()
                    }
                    # Express magnification as a virtual 20 mm segment in pixels.
                    depths.update(depth_tracker.update(
                        {side: detection['screen']
                         for side, detection in detections.items()},
                        now, frame.shape[1], frame.shape[0],
                        palm_sizes={side: 20.0 / scale if scale is not None else np.nan
                                    for side, scale in scene_scales.items()},
                    ))
                for side, detection in detections.items():
                    state = states[side]
                    update_hand_track(state, detection, now)
                    if config.depth_estimation.enabled:
                        update_hand_scene_position(
                            state, detection, mapper, depths[side],
                            frame.shape[1], frame.shape[0], now,
                            mm_per_pixel=(scene_scales[side] if scene_scales[side] is not None
                                          else np.nan),
                        )
                    drawing.draw_landmarks(
                        frame, detection['screen'], mp_hands.HAND_CONNECTIONS
                    )
                    wrist = detection['screen'].landmark[0]
                    suffix = '(locked)' if detection['label_was_stabilized'] \
                        else '{:.0f}%'.format(detection['score'] * 100.0)
                    depth = depth_estimators[side]
                    depth_text = 'Z={:.0f}mm'.format(depths[side])
                    if config.depth_estimation.enabled and not depth.calibrated:
                        depth_text += ' CAL {:.0f}%'.format(
                            depth.calibration_progress * 100.0
                        )
                    cv2.putText(
                        frame,
                        '{} / MANO_{} {} {}'.format(
                            side.upper(), side.upper(), suffix, depth_text
                        ),
                        (int(wrist.x * frame.shape[1]),
                         int(wrist.y * frame.shape[0]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 220, 80) if side == 'left'
                        else (80, 180, 255),
                        2, cv2.LINE_AA,
                    )

                retire_missing_single_hand(raw, detections, states)
                for side, state in states.items():
                    if side not in detections:
                        state.pending_landmarks = None
                    state.submit_latest(executor)

                visible = [
                    side for side, state in states.items()
                    if now - state.last_seen < resolver.track_timeout and state.vertices is not None
                ]
                for side, state in states.items():
                    if side in visible:
                        vertices = mapper.hand_vertices(
                            state.vertices, state.display_wrist, depths[side],
                            translation=state.scene_translation,
                        )
                        state.mesh.vertices = o3d.utility.Vector3dVector(vertices)
                        state.mesh.compute_vertex_normals()
                        set_geometry_visible(visualizer, state, True)
                        visualizer.update_geometry(state.mesh)
                    else:
                        set_geometry_visible(visualizer, state, False)
                visualizer.poll_events()
                visualizer.update_renderer()

                fps_frames += 1
                elapsed = now - fps_started
                if elapsed >= 0.5:
                    fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_started = now
                cv2.putText(
                    frame, 'FPS {:.1f}'.format(fps), (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    'Open3D view: {}  [1 front / 3 depth]'.format(
                        view_mode.upper()
                    ),
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (210, 210, 210), 1, cv2.LINE_AA,
                )
                if config.depth_estimation.enabled:
                    cv2.putText(
                        frame, depth_tracker.calibration_status(now),
                        (12, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (80, 255, 255), 1, cv2.LINE_AA,
                    )
                cv2.imshow('MediaPipe Hands', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (10, 13):
                    request_depth_calibration()
                elif key == ord('1'):
                    view_mode = 'front'
                    configure_view(visualizer, view_mode)
                elif key == ord('3'):
                    view_mode = 'depth'
                    configure_view(visualizer, view_mode)
                elif key in (ord('q'), 27):
                    break
        finally:
            capture.release()
            hands.close()
            cv2.destroyAllWindows()
            visualizer.destroy_window()
