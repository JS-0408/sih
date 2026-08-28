"""
tests/test_deep_matcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for DeepMatcher module.
"""

import numpy as np
from src.matching.deep_matcher import DeepMatcher


def test_deep_matcher_init() -> None:
    matcher = DeepMatcher(device="cpu")
    assert matcher.device == "cpu"


def test_deep_matcher_classical_fallback() -> None:
    matcher = DeepMatcher(device="cpu")
    img1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    kp1, kp2, matches = matcher.match(img1, img2)
    assert isinstance(kp1, list)
    assert isinstance(kp2, list)
    assert isinstance(matches, list)
