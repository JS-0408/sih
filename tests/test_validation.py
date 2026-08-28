"""
tests/test_validation.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src/metrics/validation.py (ValidationPartitioner & ValidationReport).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.validation import ValidationPartitioner, ValidationReport


@pytest.fixture
def sample_correspondences() -> tuple[np.ndarray, np.ndarray]:
    """Generate 20 synthetic point correspondences."""
    rng = np.random.default_rng(42)
    src = rng.uniform(50, 450, (20, 2)).astype(np.float32)
    # Perfect identity transform dst = src
    dst = src.copy()
    return src, dst


def test_partitioner_init():
    p = ValidationPartitioner(holdout_fraction=0.2, strategy="spatial")
    assert p.holdout_fraction == 0.2
    assert p.strategy == "spatial"


def test_partitioner_split_spatial(sample_correspondences):
    src, dst = sample_correspondences
    p = ValidationPartitioner(holdout_fraction=0.25, strategy="spatial", min_estimation_pts=8, min_validation_pts=4)

    est_src, est_dst, val_src, val_dst = p.split(src, dst)

    assert len(est_src) + len(val_src) == 20
    assert len(val_src) >= 4
    assert len(est_src) >= 8


def test_partitioner_split_random(sample_correspondences):
    src, dst = sample_correspondences
    p = ValidationPartitioner(holdout_fraction=0.25, strategy="random", seed=42)

    est_src, est_dst, val_src, val_dst = p.split(src, dst)

    assert len(est_src) + len(val_src) == 20
    assert len(val_src) == 5


def test_partitioner_insufficient_points():
    p = ValidationPartitioner(min_estimation_pts=8, min_validation_pts=4)
    src = np.zeros((5, 2))
    dst = np.zeros((5, 2))
    with pytest.raises(ValueError):
        p.split(src, dst)


def test_evaluate_perfect_identity(sample_correspondences):
    src, dst = sample_correspondences
    p = ValidationPartitioner(holdout_fraction=0.25, strategy="spatial")
    est_src, est_dst, val_src, val_dst = p.split(src, dst)

    H = np.eye(3, dtype=np.float64)
    report = p.evaluate(H, val_src, val_dst, n_estimation=len(est_src), spatial_holdout=True)

    assert report.rmse_px < 1e-4
    assert report.p95_px < 1e-4
    assert report.subpixel_claim_valid is True
    assert report.warning == ""


def test_evaluate_known_discrepancy(sample_correspondences):
    src, dst = sample_correspondences
    p = ValidationPartitioner(holdout_fraction=0.25, strategy="spatial")
    est_src, est_dst, val_src, val_dst = p.split(src, dst)

    # Shift homography by 2.5 pixels
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 2.5

    report = p.evaluate(H, val_src, val_dst, n_estimation=len(est_src), spatial_holdout=True)

    assert report.rmse_px >= 2.0
    assert report.subpixel_claim_valid is False
    assert "Sub-pixel accuracy claim is NOT independently demonstrated" in report.warning
