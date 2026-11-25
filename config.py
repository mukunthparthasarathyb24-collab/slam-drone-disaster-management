"""Runtime configuration for UAV navigation, sensors, rewards, and training."""

import os

# camera & occupancy grid
IMG_SIZE = 256
OCC_GRID = 5
UNET_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "unet.pt")
CAMERA_HFOV_DEG = 90.0

# LiDAR (matches settings.json "LidarSensor1")
LIDAR_NAME = "LidarSensor1"
LIDAR_NUM_SECTORS = 7
LIDAR_HFOV_DEG = 180.0
LIDAR_MAX_RANGE = 20.0
LIDAR_STEER_WEIGHT = 0.6  # weight on lidar vs camera steer in safety shield

# live debug window
LIVE_VIEW = False
LIVE_VIEW_CELL_PX = 70

# action: [v_forward, v_lateral, yaw_rate] in [-1, 1]
ACTION_DIM = 3

# obs: 25 (occ grid) + 3 (rel goal) + 1 (goal bearing) + 7 (lidar) + 2 (sin/cos yaw) + 3 (body vel) + 3 (prev cmd)
STATE_DIM = OCC_GRID * OCC_GRID + 3 + 1 + LIDAR_NUM_SECTORS + 2 + 3 + ACTION_DIM

# motion limits
V_MAX = 5.0        # m/s
OMEGA_MAX = 30.0   # deg/s
H_MAX = 10.0       # m (AirSim NED: z = -H_MAX)

ALPHA_SMOOTH = 0.1  # EMA smoothing on velocity commands

# rewards
R_TIME = -0.01
R_SUCCESS = 100.0
R_COLLISION = -60.0
R_TIMEOUT = -60.0
SUCCESS_RADIUS = 2.0  # m
MAX_EPISODE_STEPS = 600
YAW_ALIGN_COEFF = 0.1
JERK_COEFF = 0.15

# LiDAR obstacle penalty thresholds
LIDAR_SAFE_DIST = 6.0
LIDAR_PENALTY_COEFF = 2.0
LIDAR_SIDE_WEIGHT = 0.5  # side/rear sectors weighted less than front

GOAL_OBS_SCALE = 200.0  # meters, should be >= max goal distance

# altitude hold
VZ_MAX = 2.0
VZ_ALPHA = 0.5

# optional safety shield (keep off during RL training so policy actions aren't masked)
ENABLE_SAFETY_SHIELD = False
AVOID_DANGER_THRESHOLD = 0.15
AVOID_DANGER_HARD = 0.55
AVOID_STEER_GAIN = 1.5
AVOID_BRAKE_GAIN = 0.9
AVOID_STEER_SIGN = 1.0  # flip to -1.0 if steering direction is inverted
AVOID_EMA_ALPHA = 0.35

# AirSim connection & stepping
AIRSIM_IP = "127.0.0.1"
VEHICLE_NAME = ""
CAMERA_NAME = "0"
ACTION_DURATION = 0.3

# False = real-time stepping; True = pause sim and step by fixed ACTION_DURATION
SYNC_STEPPING = False
COMMAND_HOLD_DURATION = 5.0

# goal sampling (NED meters)
START_POSITION = (0.0, 0.0, -H_MAX)
GOAL_SAMPLING_RADIUS_MIN = 40.0
GOAL_SAMPLING_RADIUS_MAX = 110.0

# FIXED_GOAL = [80, -97.22, -0.15] # AIRSIMNH
FIXED_GOAL = [-32.57, -164.56, -0.26] # IIITS

# training
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TOTAL_TIMESTEPS = 1_000_000
