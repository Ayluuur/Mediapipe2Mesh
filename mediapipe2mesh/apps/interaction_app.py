"""Pinch-driven virtual ball interaction application."""

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
from mediapipe2mesh.interaction import (
    BallController,
    InteractiveBall,
    PinchState,
    mano_pinch_point_to_scene,
)
from mediapipe2mesh.interaction.button import DepthButton, mano_index_tip_to_scene
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
    center = np.asarray(config.button.center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('button.center must contain three finite coordinates')
    for name in ('width', 'height', 'travel', 'tip_radius'):
        value = getattr(config.button, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('button.{} must be finite and positive'.format(name))
    if not 0.0 < config.pinch.enter_distance < config.pinch.exit_distance:
        raise ValueError(
            'Require 0 < pinch.enter_distance < pinch.exit_distance'
        )
    ball_center = np.asarray(config.ball.center, dtype=float)
    if ball_center.shape != (3,) or not np.all(np.isfinite(ball_center)):
        raise ValueError('ball.center must contain three finite coordinates')
    if not np.isfinite(config.ball.radius) or config.ball.radius <= 0.0:
        raise ValueError('ball.radius must be finite and positive')
    if config.pinch.grab_tolerance < 0.0:
        raise ValueError('pinch.grab_tolerance must be nonnegative')
    if not 0.0 < config.ball.minimum_scale <= config.ball.maximum_scale:
        raise ValueError('Invalid ball scale range')
    if config.ball.scale_sensitivity <= 0.0:
        raise ValueError('ball.scale_sensitivity must be positive')
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


def  run_interaction(config):
    validate_config(config)
    hand_states = {
        side: HandState(
            side, config.mano.iterations, config.mano.pose_smoothing,
            getattr(config.tracking, 'position_filter', None),
            mirrored_input=config.camera.mirror,
        ) for side in SIDES
    }
    pinch_states = {
        side: PinchState(
            side,
            config.pinch.enter_distance,
            config.pinch.exit_distance,
            config.pinch.missing_timeout,
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
    scene_bounds = create_scene_bounds(visualizer)
    view_mode = config.viewer.initial_view
    configure_view(visualizer, view_mode)
    mapper = SceneMapper()
    ball = InteractiveBall(config.ball.radius, config.ball.center)
    ball.add_to(visualizer)
    button = DepthButton(
        config.button.center, config.button.width, config.button.height,
        config.button.travel, config.button.tip_radius,
    )
    button.add_to(visualizer)
    controller = BallController(
        ball,
        config.ball.minimum_scale,
        config.ball.maximum_scale,
        config.ball.scale_sensitivity,
    )

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    hands = create_hand_detector(mp_hands, config.detector)
    capture = open_camera(config.camera)
    next_sequence = 1
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

                # Mesh and grip point share one immutable IK snapshot per frame.
                for state in hand_states.values():
                    state.collect_result()
                raw = extract_detections(
                    results,
                    config.tracking.handedness_map,
                    config.camera.mirror,
                )
                detections = resolver.resolve(raw, hand_states, now)
                if config.depth_estimation.enabled:
                    scene_scales = {
                        side: mapper.palm_scale(
                            detection['screen'], detection['world'],
                            hand_states[side].scene_reference_keypoints,
                            frame.shape[1], frame.shape[0],
                            segment_corrections=hand_states[side].scene_segment_corrections,
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
                    hand_state = hand_states[side]
                    filtered_world = update_hand_track(
                        hand_state, detection, now
                    )
                    if config.depth_estimation.enabled:
                        update_hand_scene_position(
                            hand_state, detection, mapper, depths[side],
                            frame.shape[1], frame.shape[0], now,
                            mm_per_pixel=(scene_scales[side] if scene_scales[side] is not None
                                          else np.nan),
                        )
                    point = mano_pinch_point_to_scene(
                        hand_state, mapper, depths[side]
                    )
                    entered = pinch_states[side].update(
                        filtered_world, now, ball,
                        config.pinch.grab_tolerance,
                        next_sequence,
                        scene_point=point,
                    )
                    if entered:
                        next_sequence += 1
                    drawing.draw_landmarks(
                        frame, detection['screen'], mp_hands.HAND_CONNECTIONS
                    )
                    wrist = detection['screen'].landmark[0]
                    pinch = pinch_states[side]
                    status = ('GRAB' if pinch.controls_ball else
                              'PINCH (outside)' if pinch.pinching else 'OPEN')
                    depth = depth_estimators[side]
                    depth_text = 'Z={:.0f}mm'.format(depths[side])
                    if config.depth_estimation.enabled and not depth.calibrated:
                        depth_text += ' CAL {:.0f}%'.format(
                            depth.calibration_progress * 100.0
                        )
                    cv2.putText(
                        frame,
                        '{} / MANO_{} {} {:.1f}mm {}'.format(
                            side.upper(), side.upper(), status,
                            pinch.distance * 1000.0, depth_text,
                        ),
                        (int(wrist.x * frame.shape[1]),
                         int(wrist.y * frame.shape[0]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (80, 255, 80) if pinch.controls_ball
                        else (80, 200, 255), 2, cv2.LINE_AA,
                    )

                retire_missing_single_hand(
                    raw,
                    detections,
                    hand_states,
                    on_retire=lambda side: pinch_states[side].release(),
                )
                for side, state in hand_states.items():
                    if side not in detections:
                        state.pending_landmarks = None
                    pinch_states[side].expire_if_missing(now)
                    state.submit_latest(executor)

                controller.update(pinch_states)
                button.update({
                    side: mano_index_tip_to_scene(state, mapper, depths[side])
                    for side, state in hand_states.items()
                    if side in detections and state.vertices is not None
                })
                button.update_geometry(visualizer)
                ball.update_geometry(visualizer)
                for pinch in pinch_states.values():
                    pinch.update_geometry(visualizer)

                visible = [
                    side for side, state in hand_states.items()
                    if now - state.last_seen < resolver.track_timeout and state.vertices is not None
                ]
                for side, state in hand_states.items():
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
                    frame,
                    'FPS {:.1f}  pinch enter/exit: {:.0f}/{:.0f}mm'.format(
                        fps,
                        config.pinch.enter_distance * 1000.0,
                        config.pinch.exit_distance * 1000.0,
                    ),
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (80, 255, 80), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    'Ball scale {:.2f}x  [1 front / 3 depth]'.format(ball.scale),
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (220, 220, 220), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    'Input flip: {}  label map: {}'.format(
                        'ON' if config.camera.mirror else 'OFF',
                        config.tracking.handedness_map.upper(),
                    ),
                    (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (220, 220, 220), 1, cv2.LINE_AA,
                )
                if config.depth_estimation.enabled:
                    cv2.putText(
                        frame, depth_tracker.calibration_status(now),
                        (12, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (80, 255, 255), 1, cv2.LINE_AA,
                    )
                cv2.putText(
                    frame, 'Button: {}  presses: {}'.format(
                        'PRESSED ' + '/'.join(sorted(button.contacts))
                        if button.pressed else 'READY', button.press_count,
                    ),
                    (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (80, 255, 80) if button.pressed else (220, 220, 220),
                    2, cv2.LINE_AA,
                )
                cv2.imshow('Pinch Interaction', frame)
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
