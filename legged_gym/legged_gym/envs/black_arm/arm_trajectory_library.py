"""Scripted arm trajectory primitives and composed tasks for black_arm training."""


def _wp(yaw, p1, p2, p3, roll=0.0, payload_on=False):
    return {
        'arm_yaw_joint': yaw,
        'arm_pitch_1_joint': p1,
        'arm_pitch_2_joint': p2,
        'arm_pitch_3_joint': p3,
        'arm_roll_joint': roll,
        'payload_on': payload_on,
    }


def _concat(*segments):
    waypoints = []
    for segment in segments:
        waypoints.extend(segment)
    return waypoints


def _traj(name, segment_duration, *segments):
    return {
        'name': name,
        'segment_duration': segment_duration,
        'waypoints': _concat(*segments),
    }


# Motion primitives.
# The library is organized around three main manipulation intents:
# 1. grasp: approach the box, descend, attach payload, and lift away from the ground.
# 2. place: descend to the target area, release payload, and retract safely.
# 3. lift_hold: keep the box lifted in a conservative carrying posture.
# A fourth primitive family, transfer, bridges grasp and place so we do not have to hardcode
# every full task as a monolithic pick_xxx_place_xxx script.

GRASP_FRONT = [
    _wp(0.00, 2.70, -2.35, -0.05, 0.0, payload_on=False),
    _wp(0.05, 0.28, -1.65, -0.22, 1.0, payload_on=False),
    _wp(0.05, 1.08, -1.62, -0.50, 2.0, payload_on=True),
    _wp(0.00, 3.01, -2.41, -0.63, -1.0, payload_on=True),
]

TRANSFER_LEFT = [
    _wp(0.5, 2.56, -2.18, 0.08, 2.0, payload_on=True),
    _wp(1.5, 0.74, -1.62, -0.63, 0.0, payload_on=True),
]

TRANSFER_RIGHT = [
    _wp(-0.5, 2.56, -2.18, 0.08, -2.0, payload_on=True),
    _wp(-1.5, 0.74, -1.62, -0.63, 0.0, payload_on=True),
]

PLACE_LEFT = [
    _wp(1.45, 0.62, -1.52, -0.55, 0.8, payload_on=True),
    _wp(1.45, 0.42, -1.38, -0.42, 0.3, payload_on=True),
    _wp(1.45, 0.42, -1.38, -0.42, -0.2, payload_on=False),
    _wp(0.90, 1.10, -1.75, -0.55, -0.5, payload_on=False),
]

PLACE_RIGHT = [
    _wp(-1.45, 0.62, -1.52, -0.55, -0.8, payload_on=True),
    _wp(-1.45, 0.42, -1.38, -0.42, -0.3, payload_on=True),
    _wp(-1.45, 0.42, -1.38, -0.42, 0.2, payload_on=False),
    _wp(-0.90, 1.10, -1.75, -0.55, 0.5, payload_on=False),
]

LIFT_HOLD_FRONT = [
    _wp(0.00, 2.72, -2.28, -0.58, -0.8, payload_on=True),
    _wp(0.00, 2.68, -2.24, -0.52, -0.2, payload_on=True),
]

RETURN_FROM_HOLD = [
    _wp(0.00, 2.10, -1.85, -0.35, 0.3, payload_on=False),
]


ARM_TRAJECTORY_PRIMITIVES = {
    'grasp_front': GRASP_FRONT,
    'transfer_left': TRANSFER_LEFT,
    'transfer_right': TRANSFER_RIGHT,
    'place_left': PLACE_LEFT,
    'place_right': PLACE_RIGHT,
    'lift_hold_front': LIFT_HOLD_FRONT,
    'return_from_hold': RETURN_FROM_HOLD,
}


ARM_TRAJECTORY_LIBRARY = [
    _traj('grasp_transfer_left_place', 0.9, GRASP_FRONT, TRANSFER_LEFT, PLACE_LEFT),
    _traj('grasp_transfer_right_place', 0.9, GRASP_FRONT, TRANSFER_RIGHT, PLACE_RIGHT),
    _traj('grasp_lift_hold', 1.0, GRASP_FRONT, LIFT_HOLD_FRONT, RETURN_FROM_HOLD),
]
