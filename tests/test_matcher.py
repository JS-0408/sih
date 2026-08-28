"""
tests/test_matcher.py
~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License

Unit tests for feature matching pipelines (FLANN + RANSAC and DeepMatcher).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.matching.flann_ransac import FLANNRansacMatcher
from src.matching.deep_matcher import DeepMatcher


def test_flann_ransac_matcher_basic():
    matcher = FLANNRansacMatcher(max_keypoints=500, ratio=0.75, ransac_thresh=5.0)
    rng = np.random.default_rng(42)
    ref = rng.integers(0, 255, (256, 256), dtype=np.uint8)
    tgt = np.roll(ref, 10, axis=0)

    src_pts, dst_pts, H = matcher.match(ref, tgt)
    assert isinstance(src_pts, np.ndarray)
    assert isinstance(dst_pts, np.ndarray)


def test_deep_matcher_fallback():
    dm = DeepMatcher(device="cpu")
    rng = np.random.default_rng(42)
    ref = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    tgt = np.roll(ref, 5, axis=0)

    kp1, kp2, matches = dm.match(ref, tgt)
    assert isinstance(kp1, list)
    assert isinstance(kp2, list)
    assert isinstance(matches, list)
