"""Tests for src/matching/flann_matcher.py"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.matching.flann_matcher import FLANNMatcher
from src.processing.keypoint_detector import KeypointDetector


@pytest.fixture
def sift_descriptors() -> tuple[np.ndarray, np.ndarray]:
    """Two SIFT descriptor sets from shifted versions of the same image."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    img[30:90, 30:90] = 180   # structure for SIFT

    kd = KeypointDetector(method="SIFT", max_keypoints=200)
    kps1, d1 = kd.detect(img)

    # Shift image slightly
    shifted = np.roll(img, 5, axis=1)
    kps2, d2 = kd.detect(shifted)

    return d1, d2


class TestFLANNMatcherInit:
    def test_default_init(self) -> None:
        fm = FLANNMatcher()
        assert fm.trees == 5
        assert fm.checks == 50
        assert fm.ratio_threshold == 0.75


class TestFLANNMatcherMatch:
    def test_returns_matches(self, sift_descriptors: tuple) -> None:
        d1, d2 = sift_descriptors
        if len(d1) < 2 or len(d2) < 2:
            pytest.skip("Not enough descriptors generated for this fixture.")
        fm = FLANNMatcher()
        matches = fm.match(d1, d2)
        assert isinstance(matches, list)
        # At least some matches expected on nearly identical images
        assert len(matches) > 0

    def test_matches_sorted_by_distance(self, sift_descriptors: tuple) -> None:
        d1, d2 = sift_descriptors
        if len(d1) < 2 or len(d2) < 2:
            pytest.skip("Not enough descriptors.")
        fm = FLANNMatcher()
        matches = fm.match(d1, d2)
        if len(matches) > 1:
            distances = [m.distance for m in matches]
            assert distances == sorted(distances)

    def test_empty_descriptors_raise(self) -> None:
        fm = FLANNMatcher()
        with pytest.raises(ValueError, match="empty"):
            fm.match(np.empty((0, 128)), np.empty((0, 128)))

    def test_dimension_mismatch_raises(self) -> None:
        fm = FLANNMatcher()
        with pytest.raises(ValueError, match="dimension mismatch"):
            fm.match(
                np.random.rand(10, 128).astype(np.float32),
                np.random.rand(10, 64).astype(np.float32),
            )
