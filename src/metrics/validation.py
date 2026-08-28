"""
src/metrics/validation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License
Part of Yaazhi GeoAlign OS / ISRO SIH26166.

Independent validation partitioning and reporting.

This module enforces the scientific requirement that fitting residuals
are NEVER the sole evidence of registration quality.

The `ValidationPartitioner` splits a set of control points into:
  - An estimation set used to fit the geometric transform.
  - A held-out validation set used for independent error reporting.

The `ValidationReport` includes metrics that would satisfy a peer reviewer:
  - RMSE, median, P95, max error on held-out points
  - Spatial distribution of validation errors
  - Explicit flag indicating whether sub-pixel accuracy is genuinely demonstrated
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("Validator")


@dataclass
class ValidationReport:
    """
    Independent quality report on held-out correspondences.

    All metrics are computed on points NOT used to fit the transform.
    """
    n_estimation: int                  # points used in fitting
    n_validation: int                  # held-out points
    rmse_px: float
    median_px: float
    p95_px: float
    max_px: float
    mean_px: float
    subpixel_claim_valid: bool         # True if P95 < 1.0 px
    spatial_holdout_used: bool         # True if spatial (not random) split
    errors_px: list[float] = field(default_factory=list)   # per-point errors
    warning: str = ""

    def to_dict(self) -> dict:
        return {
            "n_estimation_points": self.n_estimation,
            "n_validation_points": self.n_validation,
            "rmse_px": round(self.rmse_px, 4),
            "median_px": round(self.median_px, 4),
            "p95_px": round(self.p95_px, 4),
            "max_px": round(self.max_px, 4),
            "mean_px": round(self.mean_px, 4),
            "subpixel_independently_verified": self.subpixel_claim_valid,
            "spatial_holdout_used": self.spatial_holdout_used,
            "warning": self.warning,
        }


class ValidationPartitioner:
    """
    Splits GCP point arrays into estimation and held-out validation sets.

    Two strategies are supported:
    - ``random``: naive random holdout (fraction ``holdout_fraction`` of points).
    - ``spatial``: reserves the outermost spatial quadrant of points as the
      holdout set — this tests generalisation to scene regions not represented
      in the fitting data.

    Parameters
    ----------
    holdout_fraction : float
        Fraction of points to withhold for validation.
    strategy : str
        ``"random"`` or ``"spatial"``.
    seed : int
        RNG seed for reproducibility of random split.
    min_estimation_pts : int
        Minimum points required in the estimation partition.
    min_validation_pts : int
        Minimum points required in the validation partition.
    """

    def __init__(
        self,
        holdout_fraction: float = 0.25,
        strategy: str = "spatial",
        seed: int = 42,
        min_estimation_pts: int = 8,
        min_validation_pts: int = 4,
    ) -> None:
        if strategy not in ("random", "spatial"):
            raise ValueError("strategy must be 'random' or 'spatial'")
        self.holdout_fraction = holdout_fraction
        self.strategy = strategy
        self.seed = seed
        self.min_estimation_pts = min_estimation_pts
        self.min_validation_pts = min_validation_pts

    def split(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Partition correspondences.

        Returns
        -------
        (est_src, est_dst, val_src, val_dst)
            Estimation and validation point arrays, each shape (N, 2).

        Raises
        ------
        ValueError
            If there are insufficient points for a meaningful split.
        """
        n = len(src_pts)
        if n < self.min_estimation_pts + self.min_validation_pts:
            raise ValueError(
                f"Need at least {self.min_estimation_pts + self.min_validation_pts} "
                f"points, got {n}."
            )

        n_val = max(self.min_validation_pts, int(n * self.holdout_fraction))
        n_est = n - n_val
        if n_est < self.min_estimation_pts:
            n_val = n - self.min_estimation_pts
            n_est = self.min_estimation_pts

        if self.strategy == "spatial":
            val_idx = self._spatial_holdout_indices(dst_pts, n_val)
        else:
            rng = np.random.default_rng(self.seed)
            val_idx = rng.choice(n, size=n_val, replace=False)

        all_idx = np.arange(n)
        est_idx = np.setdiff1d(all_idx, val_idx)

        return (
            src_pts[est_idx], dst_pts[est_idx],
            src_pts[val_idx], dst_pts[val_idx],
        )

    def evaluate(
        self,
        transform_matrix: np.ndarray,
        val_src: np.ndarray,
        val_dst: np.ndarray,
        n_estimation: int,
        spatial_holdout: bool,
    ) -> ValidationReport:
        """
        Compute independent validation error on held-out points.

        Parameters
        ----------
        transform_matrix : np.ndarray
            3×3 estimated transform (homography or padded affine).
        val_src, val_dst : np.ndarray
            Held-out source and destination point arrays, each (N, 2).
        n_estimation : int
            Number of points used to fit the transform.
        spatial_holdout : bool
            Whether spatial holdout strategy was used.
        """
        n_val = len(val_src)
        if n_val == 0:
            return ValidationReport(
                n_estimation=n_estimation, n_validation=0,
                rmse_px=float("inf"), median_px=float("inf"),
                p95_px=float("inf"), max_px=float("inf"), mean_px=float("inf"),
                subpixel_claim_valid=False,
                spatial_holdout_used=spatial_holdout,
                warning="No validation points available.",
            )

        # Project val_src through transform
        ones = np.ones((n_val, 1))
        src_h = np.hstack([val_src, ones])           # (N, 3)
        pred_h = (transform_matrix @ src_h.T).T      # (N, 3)
        # Avoid division by near-zero for affine case
        w = pred_h[:, 2:]
        w = np.where(np.abs(w) < 1e-9, 1.0, w)
        pred = pred_h[:, :2] / w

        diff = pred - val_dst
        errors = np.sqrt((diff ** 2).sum(axis=1))

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        median_e = float(np.median(errors))
        p95_e = float(np.percentile(errors, 95))
        max_e = float(np.max(errors))
        mean_e = float(np.mean(errors))

        subpixel_valid = p95_e < 1.0

        if not subpixel_valid:
            warn = (
                f"P95 validation error {p95_e:.3f} px ≥ 1.0 px. "
                "Sub-pixel accuracy claim is NOT independently demonstrated."
            )
            logger.warning(warn)
        else:
            warn = ""
            logger.info(
                f"Independent validation PASSED: RMSE={rmse:.3f} px, "
                f"P95={p95_e:.3f} px ({n_val} held-out points)."
            )

        return ValidationReport(
            n_estimation=n_estimation,
            n_validation=n_val,
            rmse_px=rmse,
            median_px=median_e,
            p95_px=p95_e,
            max_px=max_e,
            mean_px=mean_e,
            subpixel_claim_valid=subpixel_valid,
            spatial_holdout_used=spatial_holdout,
            errors_px=errors.tolist(),
            warning=warn,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_holdout_indices(dst_pts: np.ndarray, n_val: int) -> np.ndarray:
        """
        Reserve the outermost n_val points (furthest from centroid).

        This creates a test set that deliberately covers corners/edges, which
        are the most sensitive to transform extrapolation error.
        """
        centroid = dst_pts.mean(axis=0)
        dists = np.linalg.norm(dst_pts - centroid, axis=1)
        return np.argsort(dists)[-n_val:]
