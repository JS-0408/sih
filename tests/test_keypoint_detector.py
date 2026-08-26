"""Tests for src/processing/keypoint_detector.py"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.processing.keypoint_detector import KeypointDetector


@pytest.fixture
def gray_image() -> np.ndarray:
    """64×64 grayscale uint8 image with some texture."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    # Add some structure so detectors find features
    img[20:44, 20:44] = 200
    return img


@pytest.fixture
def rgb_chw_image() -> np.ndarray:
    """3×64×64 float image (rasterio channel-first layout)."""
    rng = np.random.default_rng(7)
    return rng.random((3, 64, 64)).astype(np.float32)


class TestKeypointDetectorInit:
    def test_sift_default(self) -> None:
        kd = KeypointDetector()
        assert kd.method == "SIFT"
        assert kd.max_keypoints == 5000

    def test_orb_init(self) -> None:
        kd = KeypointDetector(method="ORB", max_keypoints=500)
        assert kd.method == "ORB"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported method"):
            KeypointDetector(method="SURF")


class TestKeypointDetection:
    def test_sift_on_gray(self, gray_image: np.ndarray) -> None:
        kd = KeypointDetector(method="SIFT", max_keypoints=100)
        kps, descs = kd.detect(gray_image)
        assert isinstance(kps, list)
        assert isinstance(descs, np.ndarray)
        assert len(kps) == len(descs)
        assert len(kps) <= 100

    def test_orb_on_gray(self, gray_image: np.ndarray) -> None:
        kd = KeypointDetector(method="ORB", max_keypoints=50)
        kps, descs = kd.detect(gray_image)
        assert len(kps) <= 50

    def test_sift_on_chw_float(self, rgb_chw_image: np.ndarray) -> None:
        kd = KeypointDetector(method="SIFT")
        kps, descs = kd.detect(rgb_chw_image)
        # Should not raise — grayscale conversion handles (C,H,W)
        assert isinstance(kps, list)

    def test_empty_image_raises(self) -> None:
        kd = KeypointDetector()
        with pytest.raises(ValueError, match="empty image"):
            kd.detect(np.array([]))

    def test_keypoints_sorted_by_response(self, gray_image: np.ndarray) -> None:
        kd = KeypointDetector(method="SIFT")
        kps, _ = kd.detect(gray_image)
        if len(kps) > 1:
            responses = [k.response for k in kps]
            assert responses == sorted(responses, reverse=True)
