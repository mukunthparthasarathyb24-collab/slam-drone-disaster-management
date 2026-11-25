"""Assign AirSim segmentation IDs so obstacles share one mask color."""

import airsim

import config_collection as cfg


def apply_segmentation_ids(client: airsim.MultirotorClient):
    # reset everything to 0 (free space) before tagging obstacles
    client.simSetSegmentationObjectID("[\\w]*", 0, True)

    applied = []
    for pattern in cfg.OBSTACLE_NAME_PATTERNS:
        ok = client.simSetSegmentationObjectID(pattern, cfg.OBSTACLE_ID, True)
        applied.append((pattern, ok))
        print(f"  simSetSegmentationObjectID({pattern!r}, {cfg.OBSTACLE_ID}) -> {ok}")

    matched = sum(1 for _, ok in applied if ok)
    if matched == 0:
        print(
            "WARNING: none of OBSTACLE_NAME_PATTERNS matched any object in the "
            "scene. Open the Unreal World Outliner, check the actual object "
            "names, and update OBSTACLE_NAME_PATTERNS in config_collection.py."
        )
    return applied


def main():
    client = airsim.MultirotorClient(ip=cfg.AIRSIM_IP)
    client.confirmConnection()
    print(f"Applying segmentation IDs (obstacle ID = {cfg.OBSTACLE_ID}) ...")
    apply_segmentation_ids(client)
    print("Done.")


if __name__ == "__main__":
    main()
