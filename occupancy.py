"""Compresses U-Net probability maps into the 5x5 grid and avoidance cues."""

import numpy as np
import cv2

import config


def pool_to_grid(prob_map: np.ndarray, grid_size: int = config.OCC_GRID) -> np.ndarray:
    grid = cv2.resize(prob_map.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return grid


def flatten_grid(grid: np.ndarray) -> np.ndarray:
    return grid.flatten()


def frontal_and_global_occupancy(grid: np.ndarray):
    front_block = grid[0:2, 2:4]
    o_front = float(np.mean(front_block))
    o_global = float(np.mean(grid))
    return o_front, o_global


_COL_POS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def avoidance_steer(grid: np.ndarray):
    # focus on center rows (eye-level) and blend mean/max so thin poles still register
    center = grid[1:4, 1:4]
    danger = 0.5 * float(center.mean()) + 0.5 * float(center.max())

    eye_level_cols = grid[1:4, :].mean(axis=0)
    weight = float(eye_level_cols.sum()) + 1e-6
    obstacle_bearing = float((eye_level_cols * _COL_POS).sum() / weight)
    steer_lr = -obstacle_bearing

    return steer_lr, danger
