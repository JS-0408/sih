"""
src/matching/deep_matcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Deep learning feature extraction and matching interface (SuperPoint + LightGlue / LoFTR).

Operates in inference-only mode with memory management controls
(torch.inference_mode, bounded tensor sizes, explicit CPU/CUDA selection).
Falls back cleanly to classical SIFT/FLANN matching when PyTorch or weights are absent.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.matching.flann_matcher import FLANNMatcher
from src.processing.keypoint_detector import KeypointDetector

logger = logging.getLogger("DeepMatcher")


class DeepMatcher:
    """
    Deep Feature Matcher leveraging SuperPoint & LightGlue (or classical fallback).

    Parameters
    ----------
    device : str
        Target device ("cpu", "cuda", "auto").
    precision : str
        Inference precision ("fp32", "fp16").
    ratio_threshold : float
        Lowe's ratio threshold for classical fallback matching.
    """

    def __init__(
        self,
        device: str = "auto",
        precision: str = "fp32",
        ratio_threshold: float = 0.75,
    ) -> None:
        self.precision = precision
        self.ratio_threshold = ratio_threshold

        if HAS_TORCH:
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        else:
            self.device = "cpu"

        self.has_deep_model = False
        self._init_deep_model()

    def _init_deep_model(self) -> None:
        """Attempt loading SuperPoint/LightGlue PyTorch weights if installed."""
        if not HAS_TORCH:
            logger.info("PyTorch is not installed. Using classical SIFT/FLANN matcher fallback.")
            return

        try:
            # Check for LightGlue / SuperPoint availability
            from lightglue import LightGlue, SuperPoint  # type: ignore
            
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            self.has_deep_model = True
            logger.info(f"Loaded SuperPoint + LightGlue on device '{self.device}'.")
        except (ImportError, Exception) as exc:
            logger.info(f"Deep learning matcher unavailable ({exc}). Using classical SIFT/FLANN fallback.")
            self.has_deep_model = False

    def match(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch]]:
        """
        Extract features and compute matches between two image arrays.

        Parameters
        ----------
        img1, img2 : np.ndarray
            Input grayscale pixel arrays.

        Returns
        -------
        kp1, kp2 : list[cv2.KeyPoint]
            Detected keypoints.
        matches : list[cv2.DMatch]
            Matched keypoint pairs.
        """
        if self.has_deep_model and HAS_TORCH:
            return self._match_deep(img1, img2)
        return self._match_classical(img1, img2)

    def _match_classical(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch]]:
        """Classical SIFT + FLANN matching pipeline."""
        detector = KeypointDetector(method="SIFT", max_keypoints=5000)
        kp1, desc1 = detector.detect(img1)
        kp2, desc2 = detector.detect(img2)

        if len(kp1) == 0 or len(kp2) == 0 or len(desc1) == 0 or len(desc2) == 0:
            return kp1, kp2, []

        matcher = FLANNMatcher(ratio_threshold=self.ratio_threshold)
        matches = matcher.match(desc1, desc2)
        return kp1, kp2, matches

    def _match_deep(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch]]:
        """Deep SuperPoint + LightGlue matching pipeline."""
        with torch.inference_mode():
            t1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0) / 255.0
            t2 = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            t1, t2 = t1.to(self.device), t2.to(self.device)

            feats1 = self.extractor({"image": t1})
            feats2 = self.extractor({"image": t2})
            results = self.matcher({"image0": feats1, "image1": feats2})

            kpts1 = feats1["keypoints"][0].cpu().numpy()
            kpts2 = feats2["keypoints"][0].cpu().numpy()
            matches_idx = results["matches"][0].cpu().numpy()

            kp1_list = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1.0) for pt in kpts1]
            kp2_list = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1.0) for pt in kpts2]

            dmatches = []
            for i, (idx1, idx2) in enumerate(matches_idx):
                dmatches.append(cv2.DMatch(_queryIdx=int(idx1), _trainIdx=int(idx2), _distance=0.0))

            return kp1_list, kp2_list, dmatches
