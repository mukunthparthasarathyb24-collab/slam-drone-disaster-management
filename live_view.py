"""Debug window rendering the 5x5 occupancy grid with goal and steer markers."""

import cv2
import numpy as np

import config

_WINDOW = "UAV live probability map"


def _grid_to_heatmap(grid: np.ndarray, cell_px: int) -> np.ndarray:
    size = grid.shape[0]
    img = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (size * cell_px, size * cell_px), interpolation=cv2.INTER_NEAREST)
    for i in range(1, size):
        cv2.line(heat, (0, i * cell_px), (size * cell_px, i * cell_px), (40, 40, 40), 1)
        cv2.line(heat, (i * cell_px, 0), (i * cell_px, size * cell_px), (40, 40, 40), 1)
    return heat


def _col_to_px(col_f: float, size: int, cell_px: int) -> int:
    col_f = float(np.clip(col_f, 0.0, size - 1))
    return int((col_f + 0.5) * cell_px)


def goal_bearing_to_grid_col(bearing_deg: float) -> tuple:
    half_fov = config.CAMERA_HFOV_DEG / 2.0
    visible = abs(bearing_deg) <= half_fov
    norm = float(np.clip(bearing_deg / half_fov, -1.0, 1.0))
    col = (norm + 1.0) / 2.0 * (config.OCC_GRID - 1)
    return col, visible


def show(grid: np.ndarray, goal_col: float, goal_visible: bool, action_col: float,
         distance: float, step_count: int):
    if not config.LIVE_VIEW:
        return
    try:
        size = grid.shape[0]
        cell_px = config.LIVE_VIEW_CELL_PX
        img = _grid_to_heatmap(grid, cell_px)
        h, w = img.shape[:2]
        row_px = _col_to_px((size - 1) / 2.0, size, cell_px)

        gx = _col_to_px(goal_col, size, cell_px)
        goal_color = (0, 220, 0) if goal_visible else (0, 140, 255)
        cv2.drawMarker(img, (gx, row_px), goal_color, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, "GOAL", (max(0, gx - 18), max(14, row_px - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, goal_color, 1, cv2.LINE_AA)

        ax = _col_to_px(action_col, size, cell_px)
        ay = h - cell_px // 3
        cv2.circle(img, (ax, ay), 8, (0, 255, 255), -1)
        cv2.putText(img, "GOING", (max(0, ax - 22), h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        header = np.zeros((26, w, 3), dtype=np.uint8)
        cv2.putText(header, f"dist={distance:5.1f}m  step={step_count}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.vstack([header, img])

        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(_WINDOW, canvas)
        cv2.waitKey(1)
    except cv2.error:
        config.LIVE_VIEW = False  # disable if running headless


def close():
    try:
        cv2.destroyWindow(_WINDOW)
    except cv2.error:
        pass
