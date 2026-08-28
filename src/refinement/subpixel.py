"""
src/refinement/subpixel.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License
Part of Yaazhi GeoAlign OS / ISRO SIH26166.

Explicit sub-pixel correspondence refinement via local normalised
cross-correlation (NCC) peak fitting.

Design rationale
----------------
SIFT and ORB localise keypoints to integer pixel precision.  For
Chandrayaan-2 OHRC at 0.32 m/pixel, 1-pixel error = 0.32 m ground error.
Sub-pixel refinement reduces this to < 0.25 pixel via parabolic
interpolation of the cross-correlation peak.

Method
------
For each coarse match pair (p_src, p_dst):
1. Extract a small patch (patch_size × patch_size) around p_src in the target.
2. Extract a search patch (search_size × search_size) around p_dst in the reference.
3. Compute normalised cross-correlation (cv2.matchTemplate with TM_CCOEFF_NORMED).
4. Locate the peak NCC location to integer precision.
5. Fit a 2D parabola through the 3×3 neighbourhood of the peak to obtain
   sub-pixel (dx, dy) offset.
6. Return the refined correspondence.

Fallback: if NCC peak < min_ncc or parabola fit fails, keep coarse match.

Benchmark (see tests/test_subpixel.py):
- Known 0.3 px displacement: mean residual < 0.12 px after refinement.
- Failure rate < 5% for well-textured patches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("SubPixelRefiner")


@dataclass
class RefinedMatch:
    """Refined correspondence between two images."""
    src_x: float          # refined source x (target image)
    src_y: float
    dst_x: float          # refined destination x (reference image)
    dst_y: float
    ncc_score: float      # quality of local NCC peak
    refined: bool         # True if parabolic fit succeeded and improved result


class SubPixelRefiner:
    """
    Refines coarse pixel-precision matches to sub-pixel accuracy.

    Parameters
    ----------
    patch_size : int
        Half-width of the target patch (full = 2*patch_size+1 pixels).
    search_size : int
        Half-width of the reference search window.
    min_ncc : float
        Minimum NCC score to accept a refined match (0.0–1.0).
    max_refine_px : float
        Maximum allowed refinement displacement magnitude.
        Refinements larger than this signal a hard match error.
    """

    def __init__(
        self,
        patch_size: int = 7,
        search_size: int = 15,
        min_ncc: float = 0.5,
        max_refine_px: float = 3.0,
    ) -> None:
        self.patch_size = patch_size
        self.search_size = max(search_size, patch_size + 2)
        self.min_ncc = min_ncc
        self.max_refine_px = max_refine_px

    def refine(
        self,
        ref_gray: np.ndarray,
        tgt_gray: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        """
        Refine a set of coarse correspondences.

        Parameters
        ----------
        ref_gray : np.ndarray
            Reference image, uint8 grayscale (H, W).
        tgt_gray : np.ndarray
            Target image, uint8 grayscale (H, W).
        src_pts : np.ndarray
            Coarse source (target image) coordinates, shape (N, 2).
        dst_pts : np.ndarray
            Coarse destination (reference image) coordinates, shape (N, 2).

        Returns
        -------
        refined_src : np.ndarray  shape (N, 2)
        refined_dst : np.ndarray  shape (N, 2)
        success_mask : list[bool]  True where refinement succeeded
        """
        ref_g = self._ensure_gray(ref_gray)
        tgt_g = self._ensure_gray(tgt_gray)

        ref_h, ref_w = ref_g.shape
        tgt_h, tgt_w = tgt_g.shape

        refined_src = src_pts.copy().astype(np.float64)
        refined_dst = dst_pts.copy().astype(np.float64)
        success = []

        ps = self.patch_size
        ss = self.search_size

        for i in range(len(src_pts)):
            sx, sy = float(src_pts[i, 0]), float(src_pts[i, 1])
            dx, dy = float(dst_pts[i, 0]), float(dst_pts[i, 1])

            # Extract patch from target (source) image
            sx_i, sy_i = int(round(sx)), int(round(sy))
            dx_i, dy_i = int(round(dx)), int(round(dy))

            # Boundary check for patch
            if (sy_i - ps < 0 or sy_i + ps + 1 > tgt_h
                    or sx_i - ps < 0 or sx_i + ps + 1 > tgt_w):
                success.append(False)
                continue

            # Boundary check for search window
            if (dy_i - ss < 0 or dy_i + ss + 1 > ref_h
                    or dx_i - ss < 0 or dx_i + ss + 1 > ref_w):
                success.append(False)
                continue

            patch = tgt_g[sy_i - ps: sy_i + ps + 1, sx_i - ps: sx_i + ps + 1]
            search = ref_g[dy_i - ss: dy_i + ss + 1, dx_i - ss: dx_i + ss + 1]

            # NCC correlation
            try:
                ncc_map = cv2.matchTemplate(
                    search.astype(np.float32),
                    patch.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED,
                )
            except cv2.error:
                success.append(False)
                continue

            _, max_val, _, max_loc = cv2.minMaxLoc(ncc_map)
            if max_val < self.min_ncc:
                success.append(False)
                continue

            # Sub-pixel parabolic fit
            px, py = max_loc  # (col, row) in ncc_map
            sub_dx, sub_dy = self._parabolic_subpixel(ncc_map, px, py)

            # Convert back to reference image coordinates
            # max_loc in search window: top-left corner is (dx_i - ss, dy_i - ss)
            ref_x = (dx_i - ss) + px + ps + sub_dx
            ref_y = (dy_i - ss) + py + ps + sub_dy

            disp = np.sqrt((ref_x - dx) ** 2 + (ref_y - dy) ** 2)
            if disp > self.max_refine_px:
                success.append(False)
                continue

            refined_dst[i, 0] = ref_x
            refined_dst[i, 1] = ref_y
            success.append(True)

        return refined_src, refined_dst, success

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_gray(image: np.ndarray) -> np.ndarray:
        """Return uint8 grayscale from any array layout."""
        arr = image
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3:
            arr = arr[:, :, 0] if arr.shape[2] == 1 else cv2.cvtColor(
                arr[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY
            )
        if arr.dtype != np.uint8:
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
        return arr

    @staticmethod
    def _parabolic_subpixel(ncc_map: np.ndarray, px: int, py: int) -> tuple[float, float]:
        """
        Fit a parabola through a 3×3 neighbourhood of (px, py) in ncc_map.

        Returns fractional (dx, dy) offset within [−0.5, +0.5].
        """
        h, w = ncc_map.shape
        # Clamp to valid 3×3 neighbourhood
        if px <= 0 or px >= w - 1 or py <= 0 or py >= h - 1:
            return 0.0, 0.0

        try:
            row = ncc_map[py, px - 1: px + 2].astype(np.float64)
            col = ncc_map[py - 1: py + 2, px].astype(np.float64)

            # Parabolic peak: dx = -(f(+1) - f(-1)) / (2*(f(+1) - 2*f(0) + f(-1)))
            denom_x = 2.0 * (row[2] - 2.0 * row[1] + row[0])
            denom_y = 2.0 * (col[2] - 2.0 * col[1] + col[0])

            dx = -(row[2] - row[0]) / denom_x if abs(denom_x) > 1e-9 else 0.0
            dy = -(col[2] - col[0]) / denom_y if abs(denom_y) > 1e-9 else 0.0

            # Clamp to ±0.5
            dx = max(-0.5, min(0.5, dx))
            dy = max(-0.5, min(0.5, dy))
            return dx, dy
        except Exception:
            return 0.0, 0.0
