"""
tests/test_gcp_estimator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for GCPEstimator and ControlPoint aggregation.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.geometry.gcp_estimator import GCPEstimator, ControlPoint


def _make_pts(n: int = 30, shift_x: float = 10.0, shift_y: float = -5.0):
    """Create matched point sets with a known translation."""
    src = np.random.default_rng(42).uniform(50, 450, (n, 2)).astype(np.float32)
    dst = src + np.array([shift_x, shift_y])
    return src, dst


class TestGCPEstimatorInit:
    def test_defaults(self):
        e = GCPEstimator()
        assert e.model == "homography"
        assert e.min_inliers == 12

    def test_custom_model(self):
        e = GCPEstimator(model="affine")
        assert e.model == "affine"


class TestAddTileCorrespondences:
    def test_returns_control_points(self):
        e = GCPEstimator()
        src, dst = _make_pts(10)
        gcps = e.add_tile_correspondences(src, dst, tile_id="r0_c0")
        assert len(gcps) == 10
        assert gcps[0].tile_id == "r0_c0"
        assert isinstance(gcps[0], ControlPoint)


class TestGCPEstimate:
    def test_translation_model_success(self):
        src, dst = _make_pts(30)
        e = GCPEstimator(model="translation", min_inliers=5, max_rmse_px=15.0, min_coverage=0.05)
        gcps = e.add_tile_correspondences(src, dst)
        result = e.estimate(gcps, scene_width=512, scene_height=512)
        assert result.is_valid
        assert result.inlier_count >= 5

    def test_homography_model_success(self):
        np.random.seed(7)
        src, dst = _make_pts(40, shift_x=8.0, shift_y=3.0)
        e = GCPEstimator(model="homography", min_inliers=5, max_rmse_px=20.0, min_coverage=0.05)
        gcps = e.add_tile_correspondences(src, dst)
        result = e.estimate(gcps, scene_width=512, scene_height=512)
        assert result.inlier_count >= 5
        assert result.rmse_px < 20.0

    def test_too_few_points_returns_invalid(self):
        e = GCPEstimator(min_inliers=20)
        src, dst = _make_pts(5)
        gcps = e.add_tile_correspondences(src, dst)
        result = e.estimate(gcps, scene_width=512, scene_height=512)
        assert not result.is_valid

    def test_affine_model(self):
        src, dst = _make_pts(30, shift_x=5.0, shift_y=5.0)
        e = GCPEstimator(model="affine", min_inliers=5, max_rmse_px=20.0, min_coverage=0.05)
        gcps = e.add_tile_correspondences(src, dst)
        result = e.estimate(gcps, scene_width=512, scene_height=512)
        assert result.matrix.shape == (3, 3)
"""
tests/test_evaluator.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for PipelineEvaluator quality gate engine.
"""


import numpy as np
import pytest
from src.metrics.evaluator import PipelineEvaluator, TileStats


def _make_stats(n_ok: int, n_fail: int, inliers_per_ok: int = 20) -> list[TileStats]:
    stats = []
    for i in range(n_ok):
        stats.append(TileStats(
            tile_id=f"ok_{i}", raw_matches=30, inlier_count=inliers_per_ok,
            inlier_ratio=inliers_per_ok/30, rmse_px=0.8, status="ok",
        ))
    for i in range(n_fail):
        stats.append(TileStats(
            tile_id=f"fail_{i}", raw_matches=0, inlier_count=0,
            inlier_ratio=0.0, rmse_px=None, status="failed",
        ))
    return stats


class TestPipelineEvaluator:
    def test_all_gates_pass(self):
        stats = _make_stats(4, 0, inliers_per_ok=10)
        ev = PipelineEvaluator(min_inliers=10, max_rmse_px=5.0, min_coverage=0.05, min_tile_success_rate=0.5)
        report = ev.evaluate(stats, global_rmse_px=1.2, spatial_coverage=0.35)
        assert report.overall_pass
        assert report.successful_tiles == 4

    def test_fail_on_low_inliers(self):
        stats = _make_stats(2, 0, inliers_per_ok=3)
        ev = PipelineEvaluator(min_inliers=50, max_rmse_px=5.0, min_coverage=0.05)
        report = ev.evaluate(stats, global_rmse_px=1.0, spatial_coverage=0.5)
        assert not report.overall_pass

    def test_fail_on_high_rmse(self):
        stats = _make_stats(4, 0)
        ev = PipelineEvaluator(min_inliers=10, max_rmse_px=2.0, min_coverage=0.05)
        report = ev.evaluate(stats, global_rmse_px=15.0, spatial_coverage=0.5)
        assert not report.overall_pass

    def test_fail_on_low_coverage(self):
        stats = _make_stats(4, 0)
        ev = PipelineEvaluator(min_inliers=10, max_rmse_px=5.0, min_coverage=0.8)
        report = ev.evaluate(stats, global_rmse_px=1.0, spatial_coverage=0.05)
        assert not report.overall_pass

    def test_to_dict_serialisable(self):
        stats = _make_stats(2, 1)
        ev = PipelineEvaluator(min_inliers=5, max_rmse_px=5.0, min_coverage=0.05)
        report = ev.evaluate(stats, global_rmse_px=2.0, spatial_coverage=0.4)
        d = report.to_dict()
        assert "overall_pass" in d
        assert isinstance(d["tile_stats"], list)

    def test_systematic_residual_detection(self):
        stats = _make_stats(4, 0)
        ev = PipelineEvaluator(min_inliers=10, max_rmse_px=5.0, min_coverage=0.05)
        # Heavily skewed residuals (many near zero, one very large)
        residuals = np.array([0.1, 0.2, 0.1, 0.15, 0.1, 50.0])
        report = ev.evaluate(stats, global_rmse_px=2.0, spatial_coverage=0.4,
                             inlier_residuals=residuals)
        assert report.systematic_residual_detected
