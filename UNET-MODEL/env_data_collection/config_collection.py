"""Settings for autonomous AirSim RGB/mask data collection."""

import os

AIRSIM_IP = "127.0.0.1"
VEHICLE_NAME = ""
CAMERA_NAME = "0"
IMG_SIZE = 256

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
MASKS_DIR = os.path.join(OUTPUT_ROOT, "masks")

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obstacle_color.json")

# Unreal World Outliner patterns tagged as obstacles (everything else stays ID 0)
OBSTACLE_ID = 1
OBSTACLE_NAME_PATTERNS = [
    r"Shape_Cube.*",
    r"SM_Spotlight.*",
    r"Shape_Cylinder.*",
    r"Shape_Plane.*",
    r"Shape_Tube.*",
    r"BlockingVolume.*",
    r"SM_SuperTree.*",
    r"SM_Mobile_Trees.*",
    r"SM_Palm_01_05_G.*",
    # r"Person.*",
    # r"Fence.*",
    # r"Sign.*",
]

# sweep bounds (NED meters)
X_MIN, X_MAX = -120.0, 120.0
Y_MIN, Y_MAX = -120.0, 120.0
GRID_SPACING = 12.0
ALTITUDES = [6.0, 15.0, 30.0]  # positive heights above start (negated for NED)
WAYPOINT_JITTER = 3.0
YAW_SAMPLES_DEG = [0, 60, 120, 180, 240, 300]

# skip saving frames if drone hasn't moved/turned enough
MIN_SAVE_DIST_M = 3.0
MIN_SAVE_YAW_DEG = 20.0

TARGET_PAIRS = 8000
MOVE_VELOCITY = 6.0
DESCENT_VELOCITY = 3.0
MOVE_TIMEOUT_S = 25.0
YAW_SETTLE_S = 0.6
COLLISION_RECOVERY_ALT_M = 40.0

# transit between waypoints above tallest obstacles, then descend vertically
CRUISE_ALT_M = max(ALTITUDES) + 15.0

COLLISION_POLL_INTERVAL_S = 0.15
COLLISION_BACKOFF_M = 6.0
MAX_COLLISIONS_PER_CELL = 2

RANDOM_TOPUP = True

# hold out every Nth sweep column for spatial train/val separation
VAL_SPLIT_ENABLED = True
VAL_COLUMN_MODULO = 6
VAL_COLUMN_REMAINDER = 5
VAL_OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_val")
VAL_IMAGES_DIR = os.path.join(VAL_OUTPUT_ROOT, "images")
VAL_MASKS_DIR = os.path.join(VAL_OUTPUT_ROOT, "masks")
