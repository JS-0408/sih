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
    control point positions, in either pixel or map-coordinate space.
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
        sq_dist = np.sum(diff**2, axis=1)
        return float(math.sqrt(np.mean(sq_dist)))

    @staticmethod
    def compute_map(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        transform: Affine,
    ) -> float:
        """
        Compute map-coordinate RMSE using the raster affine transform.

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
        rmse_map : float
            RMSE in map units (degrees for geographic CRS; metres for projected CRS).
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
        sq_dist = np.sum(diff**2, axis=1)
        return float(math.sqrt(np.mean(sq_dist)))

    @staticmethod
    def compute_geo(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        transform: Affine,
    ) -> float:
        """Backward-compatible alias for map-coordinate RMSE."""
        return RMSECalculator.compute_map(predicted, ground_truth, transform)

    @staticmethod
    def summary(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        transform: Affine | None = None,
    ) -> dict[str, float]:
        """
        Return a dict with pixel RMSE and optionally map-coordinate RMSE.

        Parameters
        ----------
        predicted, ground_truth : np.ndarray
            Point arrays of shape ``(N, 2)``.
        transform : Affine | None
            If provided, also computes map-coordinate RMSE.

        Returns
        -------
        dict with keys ``"rmse_px"`` and ``"rmse_map"`` (plus legacy ``"rmse_geo"`` alias).
        """
        result: dict[str, float] = {
            "rmse_px": RMSECalculator.compute(predicted, ground_truth)
        }
        if transform is not None:
            rmse_map = RMSECalculator.compute_map(predicted, ground_truth, transform)
            result["rmse_map"] = rmse_map
            result["rmse_geo"] = rmse_map
        return result
