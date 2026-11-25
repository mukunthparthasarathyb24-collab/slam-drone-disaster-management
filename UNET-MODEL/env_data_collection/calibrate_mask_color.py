"""Calibrate the AirSim segmentation RGB color used for obstacle masks."""

import argparse
import math

import airsim
import cv2

import config_collection as cfg
from set_segmentation_ids import apply_segmentation_ids
from mask_utils import get_segmentation_bgr, dominant_colors, save_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=-10.0, help="NED z, negative = up")
    parser.add_argument("--yaw_deg", type=float, default=0.0)
    parser.add_argument("--pick_index", type=int, default=None,
                         help="Skip auto-pick and use this index from the printed "
                              "dominant-color list as the obstacle color instead.")
    args = parser.parse_args()

    client = airsim.MultirotorClient(ip=cfg.AIRSIM_IP)
    client.confirmConnection()
    client.enableApiControl(True, cfg.VEHICLE_NAME)
    client.armDisarm(True, cfg.VEHICLE_NAME)

    print("Applying segmentation IDs...")
    apply_segmentation_ids(client)

    print("Taking off and moving to calibration pose...")
    client.takeoffAsync(vehicle_name=cfg.VEHICLE_NAME).join()
    client.moveToPositionAsync(args.x, args.y, args.z, 5.0, vehicle_name=cfg.VEHICLE_NAME).join()
    client.rotateToYawAsync(args.yaw_deg, vehicle_name=cfg.VEHICLE_NAME).join()

    seg = get_segmentation_bgr(client)
    if seg is None:
        raise RuntimeError("simGetImage returned no segmentation frame. Is the sim running?")

    preview_path = "env_data_collection/calibration_preview.png"
    cv2.imwrite(preview_path, seg)
    print(f"Saved segmentation preview to {preview_path} -- open it to sanity check.")

    colors = dominant_colors(seg, top_k=8)
    print("Most frequent segmentation colors (BGR, pixel count):")
    for i, (color, count) in enumerate(colors):
        print(f"  [{i}] {color}  ({count} px)")

    background_bgr, _ = colors[0]
    if args.pick_index is not None:
        obstacle_bgr, _ = colors[args.pick_index]
    else:
        # index 0 is usually sky/ground background, index 1 is obstacles
        obstacle_bgr = colors[1][0] if len(colors) > 1 else colors[0][0]

    print(f"Background color (assumed ID 0): {background_bgr}")
    print(f"Obstacle color (assumed ID {cfg.OBSTACLE_ID}): {obstacle_bgr}")

    save_calibration(obstacle_bgr, background_bgr)
    print(f"Saved calibration to {cfg.CALIBRATION_PATH}")
    print(
        "If this looks wrong (e.g. no obstacle was actually in view), point "
        "the camera at a building/tree with --x/--y/--z/--yaw_deg and rerun, "
        "or pick the right entry manually with --pick_index."
    )

    client.armDisarm(False, cfg.VEHICLE_NAME)
    client.enableApiControl(False, cfg.VEHICLE_NAME)


if __name__ == "__main__":
    main()
