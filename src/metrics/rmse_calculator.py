"""
src/metrics/rmse_calculator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RMSE evaluation for keypoint matches in both pixel and geographic space.
"""

from __future__ import annotations

import math

import numpy as np
from rasterio.transform import Affine


class RMSECalculator:
    """
    Computes Root Mean Square Error between predicted and ground-truth
    control point positions, in either pixel or geographic (meter) space.
    """

    @staticmethod
    def compute(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """
        Compute pixel-space RMSE between two point arrays.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted point coordinates, shape ``(N, 2)`` as ``(col, row)``.
        ground_truth : np.ndarray
            Ground-truth coordinates, shape ``(N, 2)``.

        Returns
        -------
        rmse : float
            RMSE value in pixels.

        Raises
        ------
        ValueError
            If arrays are empty or have mismatched shapes.
        """
        predicted = np.asarray(predicted, dtype=np.float64)
        ground_truth = np.asarray(ground_truth, dtype=np.float64)

        if predicted.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted.shape} ≠ ground_truth {ground_truth.shape}"
            )
        if predicted.ndim != 2 or predicted.shape[1] != 2:
            raise ValueError("Arrays must have shape (N, 2).")
        if len(predicted) == 0:
            raise ValueError("Cannot compute RMSE on empty arrays.")

        diff = predicted - ground_truth
        return float(math.sqrt(np.mean(diff**2)))

    @staticmethod
    def compute_geo(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        transform: Affine,
    ) -> float:
        """
        Compute geographic RMSE in metres using the raster affine transform.

        Pixel coordinates are converted to map coordinates via the affine
        transform before computing Euclidean distances.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted pixel coordinates, shape ``(N, 2)`` as ``(col, row)``.
        ground_truth : np.ndarray
            Ground-truth pixel coordinates, shape ``(N, 2)``.
        transform : Affine
            Rasterio affine transform for pixel→map conversion.

        Returns
        -------
        rmse_m : float
            RMSE in map units (degrees if EPSG:4326; use projected CRS for metres).
        """
        predicted = np.asarray(predicted, dtype=np.float64)
        ground_truth = np.asarray(ground_truth, dtype=np.float64)

        if predicted.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted.shape} ≠ ground_truth {ground_truth.shape}"
            )

        def px_to_map(pts: np.ndarray) -> np.ndarray:
            cols, rows = pts[:, 0], pts[:, 1]
            xs = transform.c + cols * transform.a + rows * transform.b
            ys = transform.f + cols * transform.d + rows * transform.e
            return np.stack([xs, ys], axis=1)

        pred_map = px_to_map(predicted)
        gt_map = px_to_map(ground_truth)
        diff = pred_map - gt_map
        return float(math.sqrt(np.mean(diff**2)))

    @staticmethod
    def summary(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        transform: Affine | None = None,
    ) -> dict[str, float]:
        """
        Return a dict with pixel RMSE and optionally geo RMSE.

        Parameters
        ----------
        predicted, ground_truth : np.ndarray
            Point arrays of shape ``(N, 2)``.
        transform : Affine | None
            If provided, also computes geo RMSE.

        Returns
        -------
        dict with keys ``"rmse_px"`` (and ``"rmse_geo"`` when transform is given).
        """
        result: dict[str, float] = {
            "rmse_px": RMSECalculator.compute(predicted, ground_truth)
        }
        if transform is not None:
            result["rmse_geo"] = RMSECalculator.compute_geo(
                predicted, ground_truth, transform
            )
        return result
