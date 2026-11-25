"""LiDAR point-cloud processing into sector ranges and closeness features."""

import numpy as np

import config

_SECTOR_POS_CACHE = {}


def parse_point_cloud(raw_point_cloud) -> np.ndarray:
    # LidarData.point_cloud comes as a flat [x, y, z, ...] list in local body frame
    pts = np.asarray(raw_point_cloud, dtype=np.float32)
    if pts.size < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return pts.reshape(-1, 3)


def sector_ranges(points: np.ndarray, num_sectors: int = None,
                   hfov_deg: float = None, max_range: float = None) -> np.ndarray:
    num_sectors = num_sectors or config.LIDAR_NUM_SECTORS
    hfov_deg = hfov_deg if hfov_deg is not None else config.LIDAR_HFOV_DEG
    max_range = max_range if max_range is not None else config.LIDAR_MAX_RANGE

    ranges = np.full(num_sectors, max_range, dtype=np.float32)
    if points.shape[0] == 0:
        return ranges

    x, y = points[:, 0], points[:, 1]
    horiz_range = np.sqrt(x ** 2 + y ** 2)
    bearing_deg = np.degrees(np.arctan2(y, x))  # 0 = ahead, + = right

    half_fov = hfov_deg / 2.0
    in_fov = (bearing_deg >= -half_fov) & (bearing_deg <= half_fov) & (horiz_range > 0.05)
    if not np.any(in_fov):
        return ranges

    bearing_deg = bearing_deg[in_fov]
    horiz_range = np.clip(horiz_range[in_fov], 0.0, max_range)

    edges = np.linspace(-half_fov, half_fov, num_sectors + 1)
    sector_idx = np.clip(np.digitize(bearing_deg, edges) - 1, 0, num_sectors - 1)
    for i in range(num_sectors):
        in_sector = horiz_range[sector_idx == i]
        if in_sector.size > 0:
            ranges[i] = float(in_sector.min())
    return ranges


def closeness_from_ranges(ranges_m: np.ndarray, max_range: float = None) -> np.ndarray:
    # map meters to [0, 1] where 1 means right next to an obstacle
    max_range = max_range if max_range is not None else config.LIDAR_MAX_RANGE
    return 1.0 - np.clip(ranges_m / max_range, 0.0, 1.0)


def steer_from_closeness(closeness: np.ndarray):
    n = closeness.shape[0]
    if n == 0:
        return 0.0, 0.0

    sector_pos = _SECTOR_POS_CACHE.get(n)
    if sector_pos is None:
        sector_pos = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        _SECTOR_POS_CACHE[n] = sector_pos

    danger = float(closeness.max())
    weight = float(closeness.sum()) + 1e-6
    bearing = float((closeness * sector_pos).sum() / weight)
    steer_lr = -bearing
    return steer_lr, danger


def frontal_and_global_range(ranges_m: np.ndarray, max_range: float = None):
    max_range = max_range if max_range is not None else config.LIDAR_MAX_RANGE
    n = ranges_m.shape[0]
    if n == 0:
        return max_range, max_range

    # middle third of sectors as the frontal cone
    span = max(1, n // 3)
    mid = n // 2
    lo = max(0, mid - span // 2)
    hi = min(n, lo + span)
    front_min = float(ranges_m[lo:hi].min()) if hi > lo else max_range
    global_min = float(ranges_m.min())
    return front_min, global_min
