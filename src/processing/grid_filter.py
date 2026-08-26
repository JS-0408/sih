"""
src/processing/grid_filter.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Uniform spatial grid filter to prevent keypoint clustering.
Retains the highest-response keypoint per NxN grid cell.
"""

from __future__ import annotations

import cv2
import numpy as np


class GridFilter:
    """
    Spatial grid-based keypoint pruning.

    Divides the image into an NxN grid and retains only the
    strongest-response keypoint within each cell. This prevents
    feature clustering in high-texture regions and ensures uniform
    spatial coverage for robust matching.

    Parameters
    ----------
    grid_cells : int
        Number of cells per axis (produces grid_cells × grid_cells grid).
    """

    def __init__(self, grid_cells: int = 16) -> None:
        if grid_cells < 1:
            raise ValueError("grid_cells must be >= 1.")
        self.grid_cells = grid_cells

    def filter(
        self,
        keypoints: list[cv2.KeyPoint],
        image_shape: tuple[int, int],
        descriptors: np.ndarray | None = None,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """
        Prune keypoints to one best-response point per grid cell.

        Parameters
        ----------
        keypoints : list[cv2.KeyPoint]
            Input keypoints (any ordering).
        image_shape : tuple[int, int]
            ``(height, width)`` of the source image.
        descriptors : np.ndarray | None
            Descriptor matrix aligned with ``keypoints``.
            Filtered in sync when provided.

        Returns
        -------
        filtered_kp : list[cv2.KeyPoint]
            Spatially-pruned keypoints.
        filtered_desc : np.ndarray | None
            Corresponding descriptors, or ``None`` if not provided.
        """
        if not keypoints:
            return [], descriptors

        height, width = image_shape
        cell_h = height / self.grid_cells
        cell_w = width / self.grid_cells

        # Map each keypoint to a grid cell; keep highest response per cell
        best: dict[tuple[int, int], tuple[int, float]] = {}  # cell → (idx, response)
        for idx, kp in enumerate(keypoints):
            cell_r = min(int(kp.pt[1] / cell_h), self.grid_cells - 1)
            cell_c = min(int(kp.pt[0] / cell_w), self.grid_cells - 1)
            cell = (cell_r, cell_c)
            if cell not in best or kp.response > best[cell][1]:
                best[cell] = (idx, kp.response)

        selected_indices = sorted(v[0] for v in best.values())
        filtered_kp = [keypoints[i] for i in selected_indices]
        filtered_desc = (
            descriptors[selected_indices] if descriptors is not None else None
        )

        return filtered_kp, filtered_desc
