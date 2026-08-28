"""
tests/test_illumination.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src/preprocessing/illumination.py (IlluminationNormalizer).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocessing.illumination import IlluminationNormalizer, benchmark_modes


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a synthetic grayscale image with lighting gradients."""
    h, w = 256, 256
    x = np.linspace(0, 255, w)
    y = np.linspace(0, 255, h)
    xx, yy = np.meshgrid(x, y)
    img = (0.5 * xx + 0.5 * yy).astype(np.uint8)
    return img


def test_init_invalid_mode():
    with pytest.raises(ValueError):
        IlluminationNormalizer(mode="invalid_mode")  # type: ignore


def test_raw_mode(sample_image):
    norm = IlluminationNormalizer(mode="raw")
    out = norm.apply(sample_image)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_clahe_mode(sample_image):
    norm = IlluminationNormalizer(mode="clahe", clahe_clip=2.0)
    out = norm.apply(sample_image)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8
    # CLAHE should increase histogram variance
    assert out.std() >= sample_image.std() * 0.8


def test_gradient_mode(sample_image):
    norm = IlluminationNormalizer(mode="gradient", gradient_ksize=5)
    out = norm.apply(sample_image)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_log_clahe_mode(sample_image):
    norm = IlluminationNormalizer(mode="log_clahe")
    out = norm.apply(sample_image)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_multichannel_input(sample_image):
    rgb = cv2.merge([sample_image, sample_image, sample_image])
    norm = IlluminationNormalizer(mode="clahe")
    out = norm.apply(rgb)
    assert out.ndim == 2
    assert out.shape == (256, 256)


def test_benchmark_modes(sample_image):
    results = benchmark_modes(sample_image, sample_image)
    assert "raw" in results
    assert "clahe" in results
    assert "gradient" in results
    assert "log_clahe" in results
