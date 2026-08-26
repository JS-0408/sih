"""
src/matching/flann_matcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FLANN-based descriptor matching with Lowe's ratio test.
Supports both SIFT (float32) and ORB (uint8/binary) descriptors.
"""

from __future__ import annotations

import cv2
import numpy as np


class FLANNMatcher:
    """
    Fast Library for Approximate Nearest Neighbours (FLANN) matcher.

    Automatically selects the correct FLANN index type based on
    descriptor dtype (KD-Tree for float32 SIFT, LSH for binary ORB).

    Parameters
    ----------
    trees : int
        Number of KD-trees (float descriptors only).
    checks : int
        Number of recursive traversals during search.
    ratio_threshold : float
        Lowe's ratio test threshold (default 0.75).
    """

    def __init__(
        self,
        trees: int = 5,
        checks: int = 50,
        ratio_threshold: float = 0.75,
    ) -> None:
        self.trees = trees
        self.checks = checks
        self.ratio_threshold = ratio_threshold

    def _build_matcher(self, descriptors: np.ndarray) -> cv2.FlannBasedMatcher:
        """Select FLANN index params based on descriptor type."""
        if descriptors.dtype == np.uint8:
            # Binary descriptors (ORB, BRIEF, etc.) → LSH index
            index_params = {
                "algorithm": 6,   # FLANN_INDEX_LSH
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1,
            }
        else:
            # Float descriptors (SIFT, SURF) → KD-Tree index
            index_params = {"algorithm": 1, "trees": self.trees}  # FLANN_INDEX_KDTREE

        search_params = {"checks": self.checks}
        return cv2.FlannBasedMatcher(index_params, search_params)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> list[cv2.DMatch]:
        """
        Match two descriptor sets and apply Lowe's ratio test.

        Parameters
        ----------
        desc1 : np.ndarray
            Query descriptors, shape ``(N, D)``.
        desc2 : np.ndarray
            Train descriptors, shape ``(M, D)``.

        Returns
        -------
        good_matches : list[cv2.DMatch]
            Matches passing the ratio test, ordered by distance.

        Raises
        ------
        ValueError
            If either descriptor set is empty or shapes are incompatible.
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            raise ValueError("Cannot match empty descriptor sets.")
        if desc1.shape[1] != desc2.shape[1]:
            raise ValueError(
                f"Descriptor dimension mismatch: {desc1.shape[1]} vs {desc2.shape[1]}"
            )

        # Ensure correct dtype for FLANN
        if desc1.dtype != np.uint8:
            desc1 = desc1.astype(np.float32)
            desc2 = desc2.astype(np.float32)

        matcher = self._build_matcher(desc1)
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)

        good_matches: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        return sorted(good_matches, key=lambda x: x.distance)
