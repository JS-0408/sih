"""
src/geometry/gcp_estimator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ground Control Point (GCP) aggregation and optimal transformation estimation.

Collects tile-wise RANSAC inlier correspondences, applies spatial coverage
diagnostics, and estimates a single global or regional affine/homography model
from all accepted control points.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger("GCPEstimator")

TransformModel = Literal["homography", "affine", "similarity", "translation"]


@dataclass
class ControlPoint:
    """A single geospatially-tagged correspondence used for registration."""
    src_x: float      # source (target image) pixel x
    src_y: float
    dst_x: float      # destination (reference image) pixel x
    dst_y: float
    residual_px: float = 0.0
    tile_id: str = ""


@dataclass
class GCPResult:
    """Result of GCP-based global transform estimation."""
    model: TransformModel
    matrix: np.ndarray                    # 3×3 or 2×3 transform matrix
    control_points: list[ControlPoint]
    inlier_count: int
    rmse_px: float
    spatial_coverage: float               # [0, 1] fraction of scene covered
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True
    estimation_success: bool = True
    failure_reason: str | None = None


class GCPEstimator:
    """
    Aggregates tile-level control points and estimates a global spatial transform.

    Parameters
    ----------
    model : TransformModel
        Geometric model to estimate: 'homography', 'affine', 'similarity', 'translation'.
    min_inliers : int
        Minimum total inliers required to produce a valid transform.
    max_rmse_px : float
        Maximum acceptable reprojection RMSE in pixels.
    ransac_threshold : float
        Reprojection error threshold for final global RANSAC pass.
    min_coverage : float
        Minimum required spatial coverage fraction [0, 1].
    """

    def __init__(
        self,
        model: TransformModel = "homography",
        min_inliers: int = 12,
        max_rmse_px: float = 5.0,
        ransac_threshold: float = 3.0,
        min_coverage: float = 0.15,
    ) -> None:
        self.model = model
        self.min_inliers = min_inliers
        self.max_rmse_px = max_rmse_px
        self.ransac_threshold = ransac_threshold
        self.min_coverage = min_coverage

    def add_tile_correspondences(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        tile_id: str = "",
    ) -> list[ControlPoint]:
        """
        Convert raw point arrays from a tile into ControlPoint objects.

        Parameters
        ----------
        src_pts : np.ndarray
            Source (target image) coordinates, shape (N, 2).
        dst_pts : np.ndarray
            Destination (reference image) coordinates, shape (N, 2).
        tile_id : str
            Optional identifier for the tile these points came from.

        Returns
        -------
        list[ControlPoint]
        """
        gcps = []
        for s, d in zip(src_pts, dst_pts):
            gcps.append(ControlPoint(
                src_x=float(s[0]), src_y=float(s[1]),
                dst_x=float(d[0]), dst_y=float(d[1]),
                tile_id=tile_id,
            ))
        return gcps

    def estimate(
        self,
        control_points: list[ControlPoint],
        scene_width: int,
        scene_height: int,
    ) -> GCPResult:
        """
        Estimate the global spatial transformation from accumulated control points.

        Parameters
        ----------
        control_points : list[ControlPoint]
            All collected control points from all tiles.
        scene_width, scene_height : int
            Full raster dimensions for coverage calculation.

        Returns
        -------
        GCPResult
        """
        warnings: list[str] = []

        if len(control_points) < self.min_inliers:
            failure_reason = f"Only {len(control_points)} control points — need >= {self.min_inliers}."
            warnings.append(failure_reason)
            return GCPResult(
                model=self.model,
                matrix=np.eye(3, dtype=np.float64),
                control_points=control_points,
                inlier_count=0,
                rmse_px=float("inf"),
                spatial_coverage=0.0,
                warnings=warnings,
                is_valid=False,
                estimation_success=False,
                failure_reason=failure_reason,
            )

        src_pts = np.float32([[cp.src_x, cp.src_y] for cp in control_points])
        dst_pts = np.float32([[cp.dst_x, cp.dst_y] for cp in control_points])

        # Global RANSAC pass
        matrix, mask, fit_error = self._fit_model(src_pts, dst_pts)
        if fit_error:
            warnings.append(fit_error)
            return GCPResult(
                model=self.model,
                matrix=np.eye(3, dtype=np.float64),
                control_points=control_points,
                inlier_count=0,
                rmse_px=float("inf"),
                spatial_coverage=0.0,
                warnings=warnings,
                is_valid=False,
                estimation_success=False,
                failure_reason=fit_error,
            )

        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(src_pts), dtype=bool)
        inlier_src = src_pts[inlier_mask]
        inlier_dst = dst_pts[inlier_mask]

        # Compute reprojection residuals
        residuals = self._compute_residuals(matrix, inlier_src, inlier_dst)
        rmse_px = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else float("inf")

        # Tag residuals back to control points
        ri = 0
        for i, cp in enumerate(control_points):
            if inlier_mask[i]:
                cp.residual_px = float(residuals[ri])
                ri += 1

        # Spatial coverage — convex hull area of inlier dst points relative to scene
        coverage = self._coverage_fraction(inlier_dst, scene_width, scene_height)

        # Sanity checks
        if rmse_px > self.max_rmse_px:
            warnings.append(f"RMSE {rmse_px:.2f} px exceeds threshold {self.max_rmse_px} px.")
        if coverage < self.min_coverage:
            warnings.append(f"Spatial coverage {coverage:.1%} is below threshold {self.min_coverage:.1%}.")
        if not self._is_matrix_sane(matrix):
            warnings.append("Estimated transform matrix has implausible scale or determinant.")

        is_valid = (
            len(inlier_src) >= self.min_inliers
            and rmse_px <= self.max_rmse_px
            and coverage >= self.min_coverage
            and self._is_matrix_sane(matrix)
        )

        if not is_valid:
            logger.warning(f"GCP estimation quality warnings: {warnings}")
        else:
            logger.info(
                f"GCP estimation OK — inliers={len(inlier_src)}, RMSE={rmse_px:.3f} px, "
                f"coverage={coverage:.1%}"
            )

        return GCPResult(
            model=self.model,
            matrix=matrix,
            control_points=control_points,
            inlier_count=int(inlier_mask.sum()),
            rmse_px=rmse_px,
            spatial_coverage=coverage,
            warnings=warnings,
            is_valid=is_valid,
            estimation_success=True,
            failure_reason=None,
        )

    # ─── Private helpers ───────────────────────────────────────────────────

    def _fit_model(
        self, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
        """Fit the configured geometric model to src→dst point correspondences."""
        src_r = src_pts.reshape(-1, 1, 2)
        dst_r = dst_pts.reshape(-1, 1, 2)

        if self.model == "homography":
            M, mask = cv2.findHomography(src_r, dst_r, cv2.RANSAC, self.ransac_threshold)
            if M is None:
                return None, None, "Homography estimation failed."
        elif self.model == "affine":
            M, mask = cv2.estimateAffine2D(src_r, dst_r, method=cv2.RANSAC,
                                           ransacReprojThreshold=self.ransac_threshold)
            if M is None:
                return None, None, "Affine estimation failed."
            # Pad to 3×3 for uniform downstream handling
            M = np.vstack([M, [0, 0, 1]])
        elif self.model == "similarity":
            M, mask = cv2.estimateAffinePartial2D(src_r, dst_r, method=cv2.RANSAC,
                                                  ransacReprojThreshold=self.ransac_threshold)
            if M is None:
                return None, None, "Similarity estimation failed."
            M = np.vstack([M, [0, 0, 1]])
        else:  # translation
            tx = float(np.median(dst_pts[:, 0] - src_pts[:, 0]))
            ty = float(np.median(dst_pts[:, 1] - src_pts[:, 1]))
            M = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
            mask = None

        return M, mask, None

    def _compute_residuals(
        self, M: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> np.ndarray:
        """Project src_pts through M and compute Euclidean distance to dst_pts."""
        if len(src_pts) == 0:
            return np.array([])
        ones = np.ones((len(src_pts), 1))
        src_h = np.hstack([src_pts, ones])        # (N, 3)
        pred_h = (M @ src_h.T).T                  # (N, 3)
        pred = pred_h[:, :2] / pred_h[:, 2:]      # normalise
        diff = pred - dst_pts
        return np.sqrt((diff**2).sum(axis=1))

    def _coverage_fraction(
        self, pts: np.ndarray, w: int, h: int
    ) -> float:
        """Area of convex hull of pts relative to scene area [0, 1]."""
        if len(pts) < 3:
            return 0.0
        try:
            hull = cv2.convexHull(pts.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            return min(hull_area / (w * h), 1.0)
        except Exception:
            return 0.0

    def _is_matrix_sane(self, M: np.ndarray) -> bool:
        """Validate that the transform matrix has plausible scale/determinant."""
        try:
            det = abs(np.linalg.det(M[:2, :2]))
            return 0.1 < det < 20.0
        except Exception:
            return False
