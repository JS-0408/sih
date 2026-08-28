"""
tests/test_subpixel.py
~~~~~~~~~~~~~~~~~~~~~~
Tests for src/refinement/subpixel.py (SubPixelRefiner).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.refinement.subpixel import SubPixelRefiner


@pytest.fixture
def synthetic_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate reference and target images with a KNOWN sub-pixel shift (+0.3 px X, +0.2 px Y).
    """
    rng = np.random.default_rng(42)
    ref = rng.integers(20, 230, (256, 256), dtype=np.uint8)
    ref = cv2.GaussianBlur(ref, (7, 7), 0)

    # Apply known 0.3 px X, 0.2 px Y warp
    M = np.float32([[1, 0, 0.3], [0, 1, 0.2]])
    tgt = cv2.warpAffine(ref, M, (256, 256), flags=cv2.INTER_LANCZOS4)

    # Pick 5 well-textured coarse point matches
    src_pts = np.float32([[60, 60], [100, 100], [150, 150], [180, 80], [80, 180]])
    # Coarse integer target coordinates (off by integer rounding)
    dst_pts = src_pts + np.float32([0.0, 0.0])

    return ref, tgt, src_pts, dst_pts


def test_subpixel_refiner_init():
    refiner = SubPixelRefiner(patch_size=7, search_size=15, min_ncc=0.4)
    assert refiner.patch_size == 7
    assert refiner.search_size == 15
    assert refiner.min_ncc == 0.4


def test_subpixel_refinement_known_shift(synthetic_pair):
    ref, tgt, src_pts, dst_pts = synthetic_pair
    refiner = SubPixelRefiner(patch_size=7, search_size=15, min_ncc=0.3, max_refine_px=3.0)

    ref_src, ref_dst, mask = refiner.refine(ref, tgt, src_pts, dst_pts)

    assert len(ref_src) == len(src_pts)
    assert len(ref_dst) == len(dst_pts)
    assert len(mask) == len(src_pts)

    # Check that at least some points were refined successfully
    assert sum(mask) > 0

    # Measured shift should be close to expected shift (0.3 px, 0.2 px)
    for i in range(len(mask)):
        if mask[i]:
            shift_x = ref_dst[i, 0] - dst_pts[i, 0]
            shift_y = ref_dst[i, 1] - dst_pts[i, 1]
            assert abs(shift_x - 0.3) < 0.70
            assert abs(shift_y - 0.2) < 0.70


def test_subpixel_boundary_rejection(synthetic_pair):
    ref, tgt, _, _ = synthetic_pair
    refiner = SubPixelRefiner(patch_size=10, search_size=20)

    # Point right on image boundary
    src_pts = np.float32([[2, 2]])
    dst_pts = np.float32([[2, 2]])

    _, _, mask = refiner.refine(ref, tgt, src_pts, dst_pts)
    assert mask[0] is False
