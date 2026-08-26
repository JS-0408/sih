"""
src/matching/ransac_filter.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Homography estimation with RANSAC-based geometric outlier rejection.
"""

from __future__ import annotations

import cv2
import numpy as np


class RANSACFilter:
    """
    Geometric filter using RANSAC homography estimation.

    Removes geometrically inconsistent matches (outliers) by computing
    the best-fit homography between two point sets.

    Parameters
    ----------
    threshold : float
        Maximum reprojection error (pixels) to consider a match as inlier.
    max_iter : int
        Maximum number of RANSAC iterations.
    confidence : float
        Required confidence level for the estimated homography (0–1).
    min_matches : int
        Minimum number of input matches required; raises if below this.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        max_iter: int = 2000,
        confidence: float = 0.995,
        min_matches: int = 4,
    ) -> None:
        self.threshold = threshold
        self.max_iter = max_iter
        self.confidence = confidence
        self.min_matches = min_matches

    def filter(
        self,
        kp1: list[cv2.KeyPoint],
        kp2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
    ) -> tuple[list[cv2.DMatch], np.ndarray | None]:
        """
        Filter matches using RANSAC homography.

        Parameters
        ----------
        kp1 : list[cv2.KeyPoint]
            Keypoints from query image.
        kp2 : list[cv2.KeyPoint]
            Keypoints from train image.
        matches : list[cv2.DMatch]
            Putative matches from FLANN (or other) matcher.

        Returns
        -------
        inlier_matches : list[cv2.DMatch]
            Geometrically consistent (inlier) matches.
        homography : np.ndarray | None
            3×3 homography matrix, or ``None`` if estimation failed.

        Raises
        ------
        ValueError
            If fewer than ``min_matches`` matches are provided.
        """
        if len(matches) < self.min_matches:
            raise ValueError(
                f"Need ≥ {self.min_matches} matches for RANSAC; got {len(matches)}."
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            self.threshold,
            maxIters=self.max_iter,
            confidence=self.confidence,
        )

        if mask is None:
            return [], None

        inlier_mask = mask.ravel().astype(bool)
        inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]

        return inlier_matches, homography
