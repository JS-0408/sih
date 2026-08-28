"""
src/matching/flann_ransac.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License
Part of Yaazhi GeoAlign OS / ISRO SIH26166 Chandrayaan-2 Registration System.

Consolidated SIFT + FLANN + RANSAC feature matching pipeline.

Provides a single-call interface that:
  1. Detects SIFT keypoints and descriptors on both images.
  2. Runs FLANN KD-tree nearest neighbour matching with Lowe's ratio test.
  3. Applies RANSAC homography estimation to reject geometric outliers.
  4. Returns verified inlier match coordinates (src_pts, dst_pts).

Usage:
    from src.matching.flann_ransac import FLANNRansacMatcher
    matcher = FLANNRansacMatcher(max_keypoints=3000, ratio=0.75, ransac_thresh=5.0)
    src_pts, dst_pts, H = matcher.match(image_ref, image_tgt)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("FLANNRansac")


class FLANNRansacMatcher:
    """
    Single-call SIFT → FLANN → RANSAC feature matching pipeline.

    Parameters
    ----------
    max_keypoints : int
        Maximum SIFT keypoints to detect per image.
    ratio : float
        Lowe's ratio test threshold (0.0–1.0). Lower = stricter.
    ransac_thresh : float
        Maximum reprojection error (px) for RANSAC inlier classification.
    max_iter : int
        RANSAC iteration budget.
    confidence : float
        RANSAC confidence level (0–1).
    """

    def __init__(
        self,
        max_keypoints: int = 3000,
        ratio: float = 0.75,
        ransac_thresh: float = 5.0,
        max_iter: int = 2000,
        confidence: float = 0.995,
    ) -> None:
        self.max_keypoints  = max_keypoints
        self.ratio          = ratio
        self.ransac_thresh  = ransac_thresh
        self.max_iter       = max_iter
        self.confidence     = confidence

        self._sift  = cv2.SIFT_create(nfeatures=max_keypoints)
        self._flann = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5},
            {"checks": 50},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def match(
        self,
        ref_img: np.ndarray,
        tgt_img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Detect, match, and geometrically verify features between two images.

        Parameters
        ----------
        ref_img : np.ndarray
            Reference image (H×W or H×W×C), any dtype.
        tgt_img : np.ndarray
            Target image (H×W or H×W×C), any dtype.

        Returns
        -------
        src_pts : np.ndarray  shape (N, 2)  — matched target image pixel coords
        dst_pts : np.ndarray  shape (N, 2)  — matched reference image pixel coords
        H       : np.ndarray or None        — 3×3 homography matrix (None if failed)
        """
        ref_gray = self._to_gray(ref_img)
        tgt_gray = self._to_gray(tgt_img)

        kp_ref, desc_ref = self._sift.detectAndCompute(ref_gray, None)
        kp_tgt, desc_tgt = self._sift.detectAndCompute(tgt_gray, None)

        if desc_ref is None or desc_tgt is None or len(kp_ref) < 4 or len(kp_tgt) < 4:
            logger.warning("Insufficient keypoints detected for matching.")
            return np.empty((0, 2)), np.empty((0, 2)), None

        try:
            raw = self._flann.knnMatch(desc_tgt, desc_ref, k=2)
        except cv2.error as exc:
            logger.error(f"FLANN matching error: {exc}")
            return np.empty((0, 2)), np.empty((0, 2)), None

        # Lowe's ratio test
        good = [m for m, n in raw if m.distance < self.ratio * n.distance]

        if len(good) < 4:
            logger.warning(f"Only {len(good)} matches after ratio test — need >= 4.")
            return np.empty((0, 2)), np.empty((0, 2)), None

        src_pts = np.float32([kp_tgt[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.ransac_thresh,
            maxIters=self.max_iter,
            confidence=self.confidence,
        )

        if H is None or mask is None:
            logger.warning("RANSAC homography estimation failed.")
            return np.empty((0, 2)), np.empty((0, 2)), None

        inlier_mask = mask.ravel().astype(bool)
        inlier_src  = src_pts[inlier_mask].reshape(-1, 2)
        inlier_dst  = dst_pts[inlier_mask].reshape(-1, 2)

        logger.info(
            f"FLANNRansac: {len(good)} ratio matches -> {inlier_mask.sum()} RANSAC inliers"
        )
        return inlier_src, inlier_dst, H

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """Convert any raster array to uint8 grayscale."""
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.moveaxis(img, 0, -1)
        if img.ndim == 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = ((img.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
        return img
