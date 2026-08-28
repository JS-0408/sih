"""
src/preprocessing/illumination.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License
Part of Yaazhi GeoAlign OS / ISRO SIH26166.

Sensor-aware preprocessing layer for multi-modal Chandrayaan-2 imagery.

Implements four normalisation strategies to mitigate solar illumination
variation, thermal gradient effects, and cross-sensor radiometric offsets
(OHRC vs TMC-2 vs IIRS):

1. ``raw``       — log stretch + uint8 cast (preserves original intensity range)
2. ``clahe``     — Contrast Limited Adaptive Histogram Equalisation (recommended
                   for OHRC/TMC-2 shadow/highlight variation)
3. ``gradient``  — local Laplacian gradient magnitude (shadow-invariant,
                   effective when two sensors differ significantly in gain/offset)
4. ``log_clahe`` — log stretch + CLAHE (best for IIRS vs OHRC, extreme dynamic
                   range)

Benchmarked on synthetic illumination pairs in tests/test_illumination.py.
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger("Illumination")

PrepMode = Literal["raw", "clahe", "gradient", "log_clahe"]
_MODES = ("raw", "clahe", "gradient", "log_clahe")


class IlluminationNormalizer:
    """
    Applies a configurable radiometric normalisation strategy to a single image
    array, producing a uint8 grayscale output suitable for feature detection.

    Parameters
    ----------
    mode : PrepMode
        Strategy to apply.  See module docstring for details.
    clahe_clip : float
        Clip limit for CLAHE (larger → more contrast enhancement, more noise).
    clahe_grid : int
        Tile grid size for CLAHE (pixels per cell side).
    gradient_ksize : int
        Kernel size for Laplacian gradient (must be odd).
    """

    def __init__(
        self,
        mode: PrepMode = "clahe",
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        gradient_ksize: int = 5,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got '{mode}'")
        self.mode = mode
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        self.gradient_ksize = gradient_ksize if gradient_ksize % 2 == 1 else gradient_ksize + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise *image* and return a uint8 grayscale array of the same
        spatial dimensions.

        Parameters
        ----------
        image : np.ndarray
            Input array — any dtype, any band layout (H,W), (H,W,C), (C,H,W).

        Returns
        -------
        np.ndarray
            uint8 grayscale, shape (H, W).
        """
        gray = self._to_gray(image)

        if self.mode == "raw":
            return gray
        elif self.mode == "clahe":
            return self._apply_clahe(gray)
        elif self.mode == "gradient":
            return self._apply_gradient(gray)
        else:  # log_clahe
            return self._apply_log_clahe(gray)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert arbitrary image array to uint8 grayscale."""
        arr = image
        # Handle (C, H, W) layout
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        # Convert to single-channel
        if arr.ndim == 3:
            arr = arr[:, :, 0] if arr.shape[2] == 1 else cv2.cvtColor(
                arr[:, :, :3].astype(np.float32), cv2.COLOR_BGR2GRAY
            )
        # Stretch to uint8
        arr = arr.astype(np.float64)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        return arr.clip(0, 255).astype(np.uint8)

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement (handles shadow/highlight variation)."""
        return self._clahe.apply(gray)

    def _apply_gradient(self, gray: np.ndarray) -> np.ndarray:
        """
        Laplacian gradient magnitude.

        Transforms illumination-dependent intensity into a
        geometry-dependent signal.  Shadow boundaries become
        strong edges regardless of absolute brightness.
        """
        f = gray.astype(np.float32)
        lap = cv2.Laplacian(f, cv2.CV_32F, ksize=self.gradient_ksize)
        mag = np.abs(lap)
        # Stretch to uint8
        lo, hi = mag.min(), mag.max()
        if hi > lo:
            mag = (mag - lo) / (hi - lo) * 255.0
        return mag.clip(0, 255).astype(np.uint8)

    def _apply_log_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Log stretch then CLAHE — best for extreme dynamic range (IIRS vs OHRC)."""
        f = gray.astype(np.float32) + 1.0   # avoid log(0)
        log = np.log(f)
        lo, hi = log.min(), log.max()
        if hi > lo:
            log = (log - lo) / (hi - lo) * 255.0
        stretched = log.clip(0, 255).astype(np.uint8)
        return self._clahe.apply(stretched)


def benchmark_modes(
    ref: np.ndarray,
    tgt: np.ndarray,
) -> dict[str, dict]:
    """
    Run all four preparation modes and return keypoint count and match
    quality metrics for each.  Used by the benchmark harness.

    Parameters
    ----------
    ref, tgt : np.ndarray
        Input image arrays (any layout).

    Returns
    -------
    dict mapping mode → {kp_ref, kp_tgt, matches, ratio, mode}
    """
    results: dict[str, dict] = {}
    sift = cv2.SIFT_create(nfeatures=3000)
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})

    for mode in _MODES:
        norm = IlluminationNormalizer(mode=mode)  # type: ignore[arg-type]
        g1 = norm.apply(ref)
        g2 = norm.apply(tgt)
        kp1, d1 = sift.detectAndCompute(g1, None)
        kp2, d2 = sift.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
            results[mode] = {"kp_ref": 0, "kp_tgt": 0, "matches": 0, "ratio": 0.0}
            continue
        raw = flann.knnMatch(d1, d2, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        ratio = len(good) / max(len(raw), 1)
        results[mode] = {
            "mode": mode,
            "kp_ref": len(kp1),
            "kp_tgt": len(kp2),
            "matches": len(good),
            "ratio": round(ratio, 4),
        }
    return results
