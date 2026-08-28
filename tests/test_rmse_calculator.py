"""Tests for src/metrics/rmse_calculator.py"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine, from_bounds

from src.metrics.rmse_calculator import RMSECalculator


@pytest.fixture
def perfect_match() -> tuple[np.ndarray, np.ndarray]:
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    return pts, pts.copy()


@pytest.fixture
def offset_match() -> tuple[np.ndarray, np.ndarray]:
    pred = np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 0.0]])
    gt   = np.array([[3.0, 0.0], [1.0, 1.0], [4.0, 0.0]])
    return pred, gt


class TestRMSEPixel:
    def test_zero_error_on_perfect(self, perfect_match: tuple) -> None:
        pred, gt = perfect_match
        assert RMSECalculator.compute(pred, gt) == pytest.approx(0.0)

    def test_known_rmse(self, offset_match: tuple) -> None:
        pred, gt = offset_match
        # point-wise euclidean distances = [3, 0, 0] -> RMSE = sqrt(mean([9,0,0])) = sqrt(3)
        result = RMSECalculator.compute(pred, gt)
        assert result == pytest.approx(1.7320508, rel=1e-4)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            RMSECalculator.compute(
                np.array([[0.0, 0.0]]),
                np.array([[0.0, 0.0], [1.0, 1.0]]),
            )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            RMSECalculator.compute(np.empty((0, 2)), np.empty((0, 2)))

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            RMSECalculator.compute(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
            )


class TestRMSEGeo:
    def test_geo_rmse_returns_float(self, offset_match: tuple) -> None:
        pred, gt = offset_match
        transform = from_bounds(0, 0, 1, 1, 100, 100)
        result = RMSECalculator.compute_geo(pred, gt, transform)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_map_rmse_matches_pixel_for_identity_transform(self, offset_match: tuple) -> None:
        pred, gt = offset_match
        transform = Affine.identity()
        assert RMSECalculator.compute_map(pred, gt, transform) == pytest.approx(
            RMSECalculator.compute(pred, gt)
        )


class TestRMSESummary:
    def test_summary_no_transform(self, offset_match: tuple) -> None:
        pred, gt = offset_match
        out = RMSECalculator.summary(pred, gt)
        assert "rmse_px" in out
        assert "rmse_geo" not in out

    def test_summary_with_transform(self, offset_match: tuple) -> None:
        pred, gt = offset_match
        transform = from_bounds(0, 0, 1, 1, 100, 100)
        out = RMSECalculator.summary(pred, gt, transform)
        assert "rmse_px" in out
        assert "rmse_map" in out
        assert "rmse_geo" in out
