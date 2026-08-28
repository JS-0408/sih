"""
src/metrics/evaluator.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Automated quality gate engine for the registration pipeline.

Combines per-tile statistics, global GCP metrics, spatial coverage diagnostics,
and residual structure analysis into a single acceptance decision and JSON report.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Evaluator")


@dataclass
class TileStats:
    tile_id: str
    raw_matches: int
    inlier_count: int
    inlier_ratio: float
    rmse_px: float | None
    status: str            # "ok", "failed", "skipped"


@dataclass
class EvaluationReport:
    """Full quality evaluation report for one pipeline run."""

    overall_pass: bool
    status_message: str

    # Tile-level
    total_tiles: int
    successful_tiles: int
    failed_tiles: int

    # Feature-level
    total_raw_matches: int
    total_inliers: int
    global_inlier_ratio: float

    # Geometry
    global_rmse_px: float
    spatial_coverage: float
    systematic_residual_detected: bool

    # Quality gate thresholds used
    thresholds: dict[str, Any]

    # Per-tile detail
    tile_stats: list[TileStats] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tile_stats"] = [asdict(t) for t in self.tile_stats]
        return d


class PipelineEvaluator:
    """
    Aggregates tile and GCP metrics and enforces quality gates.

    Parameters
    ----------
    min_inliers : int
        Minimum total inlier control points required.
    max_rmse_px : float
        Maximum acceptable global RMSE in pixels.
    min_coverage : float
        Minimum spatial coverage fraction [0, 1].
    min_tile_success_rate : float
        Minimum fraction of tiles that must succeed [0, 1].
    """

    def __init__(
        self,
        min_inliers: int = 12,
        max_rmse_px: float = 5.0,
        min_coverage: float = 0.15,
        min_tile_success_rate: float = 0.5,
    ) -> None:
        self.min_inliers = min_inliers
        self.max_rmse_px = max_rmse_px
        self.min_coverage = min_coverage
        self.min_tile_success_rate = min_tile_success_rate

    def evaluate(
        self,
        tile_stats: list[TileStats],
        global_rmse_px: float,
        spatial_coverage: float,
        inlier_residuals: np.ndarray | None = None,
    ) -> EvaluationReport:
        """
        Run all quality gates and return a comprehensive evaluation report.

        Parameters
        ----------
        tile_stats : list[TileStats]
            Per-tile match and RANSAC statistics.
        global_rmse_px : float
            Global reprojection RMSE after GCP-based estimation.
        spatial_coverage : float
            Fraction of scene area covered by inlier control points.
        inlier_residuals : np.ndarray | None
            Per-inlier residual distances for systematic-error detection.

        Returns
        -------
        EvaluationReport
        """
        warnings: list[str] = []

        # Aggregate tile metrics
        total_tiles = len(tile_stats)
        successful_tiles = sum(1 for t in tile_stats if t.status == "ok")
        failed_tiles = total_tiles - successful_tiles
        total_raw = sum(t.raw_matches for t in tile_stats)
        total_inliers = sum(t.inlier_count for t in tile_stats)
        global_inlier_ratio = total_inliers / total_raw if total_raw > 0 else 0.0
        tile_success_rate = successful_tiles / total_tiles if total_tiles > 0 else 0.0

        # Detect systematic residual spatial bias (simple variance test)
        systematic = self._detect_systematic_residuals(inlier_residuals)
        if systematic:
            warnings.append(
                "Systematic residual pattern detected — global homography may be insufficient. "
                "Consider affine model or local piecewise correction."
            )

        # Quality gate verdicts
        gate_inliers = total_inliers >= self.min_inliers
        gate_rmse = global_rmse_px <= self.max_rmse_px
        gate_coverage = spatial_coverage >= self.min_coverage
        gate_tiles = tile_success_rate >= self.min_tile_success_rate

        if not gate_inliers:
            warnings.append(
                f"Total inliers {total_inliers} < minimum {self.min_inliers}."
            )
        if not gate_rmse:
            warnings.append(
                f"Global RMSE {global_rmse_px:.2f} px exceeds limit {self.max_rmse_px} px."
            )
        if not gate_coverage:
            warnings.append(
                f"Spatial coverage {spatial_coverage:.1%} below minimum {self.min_coverage:.1%}."
            )
        if not gate_tiles:
            warnings.append(
                f"Tile success rate {tile_success_rate:.1%} below minimum {self.min_tile_success_rate:.1%}."
            )

        overall_pass = gate_inliers and gate_rmse and gate_coverage and gate_tiles

        thresholds = {
            "min_inliers": self.min_inliers,
            "max_rmse_px": self.max_rmse_px,
            "min_coverage": self.min_coverage,
            "min_tile_success_rate": self.min_tile_success_rate,
        }

        status_msg = "PASS" if overall_pass else f"FAIL — {'; '.join(warnings)}"
        logger.info(
            f"Evaluation: {status_msg} | "
            f"tiles={successful_tiles}/{total_tiles} | "
            f"inliers={total_inliers} | RMSE={global_rmse_px:.3f} px | "
            f"coverage={spatial_coverage:.1%}"
        )

        return EvaluationReport(
            overall_pass=overall_pass,
            status_message=status_msg,
            total_tiles=total_tiles,
            successful_tiles=successful_tiles,
            failed_tiles=failed_tiles,
            total_raw_matches=total_raw,
            total_inliers=total_inliers,
            global_inlier_ratio=round(global_inlier_ratio, 4),
            global_rmse_px=round(global_rmse_px, 4),
            spatial_coverage=round(spatial_coverage, 4),
            systematic_residual_detected=systematic,
            thresholds=thresholds,
            tile_stats=tile_stats,
            warnings=warnings,
        )

    # ─── Private helpers ───────────────────────────────────────────────────

    def _detect_systematic_residuals(
        self, residuals: np.ndarray | None
    ) -> bool:
        """Return True if residuals show non-random spatial structure (high skew)."""
        if residuals is None or len(residuals) < 5:
            return False
        mean = float(np.mean(residuals))
        std = float(np.std(residuals))
        if std == 0:
            return False
        skewness = float(np.mean(((residuals - mean) / std) ** 3))
        # Skew > 1.5 suggests outlier cluster or systematic directional error
        return abs(skewness) > 1.5
