"""Autonomous lawnmower flight script to collect RGB/mask pairs in AirSim."""

import argparse
import glob
import math
import os
import random
import time

import cv2
import numpy as np
import airsim

import config_collection as cfg
from set_segmentation_ids import apply_segmentation_ids
from mask_utils import get_scene_bgr, get_segmentation_bgr, load_calibration, seg_to_mask


def frange(start, stop, step):
    vals = []
    x = start
    while x <= stop + 1e-6:
        vals.append(round(x, 3))
        x += step
    if not vals:
        vals = [start]
    return vals


def yaw_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def col_index_for_x(x):
    return int(round((x - cfg.X_MIN) / cfg.GRID_SPACING))


def is_val_column(col_idx):
    if not cfg.VAL_SPLIT_ENABLED:
        return False
    return (col_idx % cfg.VAL_COLUMN_MODULO) == cfg.VAL_COLUMN_REMAINDER


def generate_lawnmower_waypoints():
    xs = frange(cfg.X_MIN, cfg.X_MAX, cfg.GRID_SPACING)
    waypoints = []
    for z_up in cfg.ALTITUDES:
        z_ned = -abs(z_up)
        for col_idx, x in enumerate(xs):
            ys = frange(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_SPACING)
            if col_idx % 2 == 1:
                ys = list(reversed(ys))
            for y in ys:
                jx = x + random.uniform(-cfg.WAYPOINT_JITTER, cfg.WAYPOINT_JITTER)
                jy = y + random.uniform(-cfg.WAYPOINT_JITTER, cfg.WAYPOINT_JITTER)
                waypoints.append({"x": jx, "y": jy, "z": z_ned, "col_idx": col_idx})
    return waypoints


def random_waypoint():
    x = random.uniform(cfg.X_MIN, cfg.X_MAX)
    y = random.uniform(cfg.Y_MIN, cfg.Y_MAX)
    z_up = random.choice(cfg.ALTITUDES)
    return {"x": x, "y": y, "z": -abs(z_up), "col_idx": col_index_for_x(x)}


def cell_key(x, y):
    return (round(x / cfg.GRID_SPACING), round(y / cfg.GRID_SPACING))


class Collector:
    def __init__(self, target_pairs):
        self.target_pairs = target_pairs
        self.client = airsim.MultirotorClient(ip=cfg.AIRSIM_IP)
        self.client.confirmConnection()
        self.client.enableApiControl(True, cfg.VEHICLE_NAME)
        self.client.armDisarm(True, cfg.VEHICLE_NAME)

        # second connection needed because msgpackrpc blocks per client during .join()
        self.watch_client = airsim.MultirotorClient(ip=cfg.AIRSIM_IP)
        self.watch_client.confirmConnection()
        self._last_collision_ts = 0

        self.obstacle_bgr, _ = load_calibration()

        for d in (cfg.IMAGES_DIR, cfg.MASKS_DIR, cfg.VAL_IMAGES_DIR, cfg.VAL_MASKS_DIR):
            os.makedirs(d, exist_ok=True)

        self.frame_idx = self._next_frame_index()
        self.saved_count = self._count_existing()
        self.last_saved_pos = None
        self.last_saved_yaw = None
        self.blocked_cells = {}  # cell_key -> collision count

    def _next_frame_index(self):
        existing = glob.glob(os.path.join(cfg.IMAGES_DIR, "frame_*.png")) + \
                   glob.glob(os.path.join(cfg.VAL_IMAGES_DIR, "frame_*.png"))
        best = 0
        for path in existing:
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                n = int(name.split("_")[1])
                best = max(best, n)
            except (IndexError, ValueError):
                continue
        return best + 1

    def _count_existing(self):
        n_train = len(glob.glob(os.path.join(cfg.IMAGES_DIR, "frame_*.png")))
        n_val = len(glob.glob(os.path.join(cfg.VAL_IMAGES_DIR, "frame_*.png")))
        return n_train + n_val

    def _get_pose(self):
        state = self.client.getMultirotorState(vehicle_name=cfg.VEHICLE_NAME)
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        _, _, yaw_rad = airsim.to_eularian_angles(orientation)
        return np.array([pos.x_val, pos.y_val, pos.z_val]), math.degrees(yaw_rad)

    def _should_save(self, pos, yaw_deg):
        if self.last_saved_pos is None:
            return True
        dist = float(np.linalg.norm(pos - self.last_saved_pos))
        dyaw = yaw_diff_deg(yaw_deg, self.last_saved_yaw)
        return dist >= cfg.MIN_SAVE_DIST_M or dyaw >= cfg.MIN_SAVE_YAW_DEG

    def _save_pair(self, col_idx):
        scene = get_scene_bgr(self.client)
        seg = get_segmentation_bgr(self.client)
        if scene is None or seg is None:
            print("  [warn] missing camera frame, skipping capture")
            return False

        mask = seg_to_mask(seg, self.obstacle_bgr)

        val = is_val_column(col_idx)
        images_dir = cfg.VAL_IMAGES_DIR if val else cfg.IMAGES_DIR
        masks_dir = cfg.VAL_MASKS_DIR if val else cfg.MASKS_DIR

        fname = f"frame_{self.frame_idx:06d}.png"
        cv2.imwrite(os.path.join(images_dir, fname), scene)
        cv2.imwrite(os.path.join(masks_dir, fname), mask)

        self.frame_idx += 1
        self.saved_count += 1
        return True

    def _watch_collision(self, task, timeout_s):
        deadline = time.time() + timeout_s
        info = None
        while time.time() < deadline:
            info = self.watch_client.simGetCollisionInfo(vehicle_name=cfg.VEHICLE_NAME)
            if info.has_collided and info.time_stamp != self._last_collision_ts:
                self._last_collision_ts = info.time_stamp
                try:
                    self.watch_client.cancelLastTask(vehicle_name=cfg.VEHICLE_NAME)
                except Exception:
                    pass
                break
            info = None
            time.sleep(cfg.COLLISION_POLL_INTERVAL_S)
        try:
            task.join()
        except Exception:
            pass
        return info

    def _back_away_from_collision(self, info):
        n = np.array([info.normal.x_val, info.normal.y_val, info.normal.z_val])
        norm = np.linalg.norm(n)
        n = (n / norm) if norm > 1e-3 else np.array([0.0, 0.0, -1.0])

        pos, _ = self._get_pose()
        target = pos + n * cfg.COLLISION_BACKOFF_M
        target[2] = min(target[2], -abs(cfg.COLLISION_RECOVERY_ALT_M))  # NED: more negative is higher

        print(f"  [collision] backing off toward {tuple(round(v, 1) for v in target)}")
        task = self.client.moveToPositionAsync(
            float(target[0]), float(target[1]), float(target[2]), 3.0,
            timeout_sec=10.0, vehicle_name=cfg.VEHICLE_NAME,
        )
        self._watch_collision(task, 10.0)

    def _register_collision(self, x, y):
        key = cell_key(x, y)
        count = self.blocked_cells.get(key, 0) + 1
        self.blocked_cells[key] = count
        if count >= cfg.MAX_COLLISIONS_PER_CELL:
            print(f"  [collision] cell {key} hit {count}x, blacklisting it for the rest of this run")

    def visit_waypoint(self, wp):
        if self.saved_count >= self.target_pairs:
            return

        key = cell_key(wp["x"], wp["y"])
        if self.blocked_cells.get(key, 0) >= cfg.MAX_COLLISIONS_PER_CELL:
            print(f"  [skip] {key} is blacklisted after repeated collisions")
            return

        cruise_z = -abs(cfg.CRUISE_ALT_M)

        # climb to cruise altitude before moving horizontally
        pos, _ = self._get_pose()
        if pos[2] > cruise_z + 2.0:
            task = self.client.moveToZAsync(cruise_z, cfg.MOVE_VELOCITY, vehicle_name=cfg.VEHICLE_NAME)
            info = self._watch_collision(task, cfg.MOVE_TIMEOUT_S)
            if info is not None:
                self._register_collision(pos[0], pos[1])
                self._back_away_from_collision(info)
                return

        # transit at cruise altitude
        task = self.client.moveToPositionAsync(
            wp["x"], wp["y"], cruise_z, cfg.MOVE_VELOCITY,
            timeout_sec=cfg.MOVE_TIMEOUT_S, vehicle_name=cfg.VEHICLE_NAME,
        )
        info = self._watch_collision(task, cfg.MOVE_TIMEOUT_S)
        if info is not None:
            self._register_collision(wp["x"], wp["y"])
            self._back_away_from_collision(info)
            return

        # descend to target altitude slowly and abort on collision
        task = self.client.moveToPositionAsync(
            wp["x"], wp["y"], wp["z"], cfg.DESCENT_VELOCITY,
            timeout_sec=cfg.MOVE_TIMEOUT_S, vehicle_name=cfg.VEHICLE_NAME,
        )
        info = self._watch_collision(task, cfg.MOVE_TIMEOUT_S)
        if info is not None:
            self._register_collision(wp["x"], wp["y"])
            self._back_away_from_collision(info)
            return

        for yaw in cfg.YAW_SAMPLES_DEG:
            if self.saved_count >= self.target_pairs:
                return
            try:
                self.client.rotateToYawAsync(
                    float(yaw), timeout_sec=5.0, vehicle_name=cfg.VEHICLE_NAME
                ).join()
            except Exception as e:
                print(f"  [warn] rotate failed: {e}")
                continue
            time.sleep(cfg.YAW_SETTLE_S)

            collision = self.watch_client.simGetCollisionInfo(vehicle_name=cfg.VEHICLE_NAME)
            if collision.has_collided and collision.time_stamp != self._last_collision_ts:
                self._last_collision_ts = collision.time_stamp
                self._register_collision(wp["x"], wp["y"])
                self._back_away_from_collision(collision)
                return

            pos, yaw_deg = self._get_pose()
            if not self._should_save(pos, yaw_deg):
                continue

            if self._save_pair(wp["col_idx"]):
                self.last_saved_pos = pos
                self.last_saved_yaw = yaw_deg
                if self.saved_count % 50 == 0:
                    print(f"  saved {self.saved_count}/{self.target_pairs} pairs "
                          f"(pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) yaw={yaw_deg:.0f})")

    def run(self):
        print("Applying segmentation IDs...")
        apply_segmentation_ids(self.client)

        print("Taking off...")
        self.client.takeoffAsync(vehicle_name=cfg.VEHICLE_NAME).join()

        waypoints = generate_lawnmower_waypoints()
        print(f"Generated {len(waypoints)} lawnmower waypoints across "
              f"{len(cfg.ALTITUDES)} altitude layer(s). Target pairs: {self.target_pairs}. "
              f"Already have {self.saved_count} on disk.")

        for i, wp in enumerate(waypoints):
            if self.saved_count >= self.target_pairs:
                break
            print(f"Waypoint {i + 1}/{len(waypoints)}  x={wp['x']:.1f} y={wp['y']:.1f} z={wp['z']:.1f}")
            self.visit_waypoint(wp)

        if cfg.RANDOM_TOPUP and self.saved_count < self.target_pairs:
            print("Lawnmower sweep complete, starting randomized top-up pass...")
            max_extra_waypoints = (self.target_pairs - self.saved_count) * 3 + 50
            tries = 0
            while self.saved_count < self.target_pairs and tries < max_extra_waypoints:
                tries += 1
                wp = random_waypoint()
                self.visit_waypoint(wp)

        print(f"Done. Saved {self.saved_count} total pairs "
              f"({len(glob.glob(os.path.join(cfg.IMAGES_DIR, 'frame_*.png')))} train / "
              f"{len(glob.glob(os.path.join(cfg.VAL_IMAGES_DIR, 'frame_*.png')))} val).")

        self.client.armDisarm(False, cfg.VEHICLE_NAME)
        self.client.enableApiControl(False, cfg.VEHICLE_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_pairs", type=int, default=cfg.TARGET_PAIRS)
    args = parser.parse_args()

    collector = Collector(target_pairs=args.target_pairs)
    collector.run()


if __name__ == "__main__":
    main()
