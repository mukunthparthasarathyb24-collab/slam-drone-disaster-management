"""Step reward calculation for body-frame velocity control."""

import math
import numpy as np

import config
from occupancy import frontal_and_global_occupancy


def yaw_alignment_error_deg(current_yaw_deg: float, dx: float, dy: float) -> float:
    # AirSim uses NED (x=North, y=East); wrap error into [0, 180]
    theta_target = math.degrees(math.atan2(dy, dx))
    diff = (theta_target - current_yaw_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


def compute_reward(
    d_prev: float,
    d_curr: float,
    current_yaw_deg: float,
    dx: float,
    dy: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    occupancy_grid: np.ndarray,
    collided: bool,
    timed_out: bool,
    dt: float = config.ACTION_DURATION,
    lidar_front_m: float = config.LIDAR_MAX_RANGE,
    lidar_global_m: float = config.LIDAR_MAX_RANGE,
):
    # normalize progress by max distance covered over actual step dt
    max_step = config.V_MAX * max(dt, 1e-3)
    r_progress = float(np.clip((d_prev - d_curr) / max_step, -1.0, 1.0))

    # linear obstacle penalty inside LIDAR_SAFE_DIST (lower weight for side/rear)
    barrier_front = max(0.0, 1.0 - lidar_front_m / config.LIDAR_SAFE_DIST)
    barrier_any = max(0.0, 1.0 - lidar_global_m / config.LIDAR_SAFE_DIST)
    r_obs = -config.LIDAR_PENALTY_COEFF * max(barrier_front, config.LIDAR_SIDE_WEIGHT * barrier_any)

    e_theta = yaw_alignment_error_deg(current_yaw_deg, dx, dy)
    r_yaw = -config.YAW_ALIGN_COEFF * e_theta / 180.0

    # penalize abrupt changes in consecutive raw actions
    r_jerk = -config.JERK_COEFF * float(np.linalg.norm(action - prev_action))

    r_time = config.R_TIME

    r_terminal = 0.0
    if d_curr < config.SUCCESS_RADIUS:
        r_terminal = config.R_SUCCESS
    elif collided:
        r_terminal = config.R_COLLISION
    elif timed_out:
        r_terminal = config.R_TIMEOUT

    total = r_progress + r_obs + r_yaw + r_jerk + r_time + r_terminal

    # occupancy metrics are tracked for logging only
    o_front, o_global = frontal_and_global_occupancy(occupancy_grid)
    info = dict(
        r_progress=r_progress,
        r_obs=r_obs,
        r_yaw=r_yaw,
        r_jerk=r_jerk,
        r_time=r_time,
        r_terminal=r_terminal,
        o_front=o_front,
        o_global=o_global,
        e_theta=e_theta,
    )
    return total, info
