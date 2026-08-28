from __future__ import annotations

import cv2
import pytest

from src.matching.ransac_filter import RANSACFilter


def _kp(x: float, y: float) -> cv2.KeyPoint:
    return cv2.KeyPoint(float(x), float(y), 1)


def _match(i: int) -> cv2.DMatch:
    return cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.1)


def test_ransac_raises_when_homography_fails(monkeypatch) -> None:
    kp1 = [_kp(0, 0), _kp(1, 0), _kp(2, 0), _kp(3, 0)]
    kp2 = [_kp(0, 1), _kp(1, 1), _kp(2, 1), _kp(3, 1)]
    matches = [_match(i) for i in range(4)]

    def _fail(*args, **kwargs):
        return None, None

    monkeypatch.setattr(cv2, "findHomography", _fail)

    with pytest.raises(ValueError, match="failed"):
        RANSACFilter().filter(kp1, kp2, matches)
