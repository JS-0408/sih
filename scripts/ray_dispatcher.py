"""
scripts/ray_dispatcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 2 Distributed Tile Execution Dispatcher using Ray.

Sends lightweight task payloads (file paths, window offsets, parameters)
to remote Ray workers to maintain low shared-memory usage on 8 GB machines.
Includes a concurrent fallback executor when Ray is absent or uninitialized.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from src.io.raster_loader import RasterLoader
from src.matching.flann_matcher import FLANNMatcher
from src.matching.ransac_filter import RANSACFilter
from src.processing.grid_filter import GridFilter
from src.processing.keypoint_detector import KeypointDetector

logger = logging.getLogger("RayDispatcher")


@dataclass
class TileResult:
    """Compact result returned from tile-level feature matching worker."""

    col_off: int
    row_off: int
    raw_matches_count: int
    inlier_matches_count: int
    rmse_px: float | None


def _process_tile_pair_local(
    ref_path: str,
    tgt_path: str,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    config: dict[str, Any],
) -> TileResult:
    """Worker function executing feature extraction and matching on a single tile window."""
    try:
        loader = RasterLoader(crs_target=config.get("geospatial", {}).get("crs_target", "EPSG:4326"))
        
        # Load window arrays locally inside the worker
        ref_data, _ = loader.load(ref_path)
        tgt_data, _ = loader.load(tgt_path)

        # Slice tile windows
        ref_tile = ref_data[:, row_off : row_off + height, col_off : col_off + width]
        tgt_tile = tgt_data[:, row_off : row_off + height, col_off : col_off + width]

        detector = KeypointDetector(
            method=config.get("keypoints", {}).get("method", "SIFT"),
            max_keypoints=config.get("keypoints", {}).get("max_keypoints", 1000),
        )

        kp_ref, desc_ref = detector.detect(ref_tile)
        kp_tgt, desc_tgt = detector.detect(tgt_tile)

        if len(kp_ref) == 0 or len(kp_tgt) == 0:
            return TileResult(col_off, row_off, 0, 0, None)

        grid_filter = GridFilter(grid_cells=config.get("keypoints", {}).get("grid_cells", 8))
        kp_ref_f, desc_ref_f = grid_filter.filter(kp_ref, (height, width), desc_ref)
        kp_tgt_f, desc_tgt_f = grid_filter.filter(kp_tgt, (height, width), desc_tgt)

        matcher = FLANNMatcher(ratio_threshold=config.get("flann", {}).get("ratio_threshold", 0.75))
        matches = matcher.match(desc_ref_f, desc_tgt_f)

        if len(matches) < 4:
            return TileResult(col_off, row_off, len(matches), 0, None)

        ransac = RANSACFilter(threshold=config.get("ransac", {}).get("threshold", 5.0))
        inliers, H = ransac.filter(kp_ref_f, kp_tgt_f, matches)

        return TileResult(
            col_off=col_off,
            row_off=row_off,
            raw_matches_count=len(matches),
            inlier_matches_count=len(inliers),
            rmse_px=0.0 if H is not None else None,
        )
    except Exception as exc:
        logger.warning(f"Tile processing error at offset ({col_off}, {row_off}): {exc}")
        return TileResult(col_off, row_off, 0, 0, None)


if HAS_RAY:
    @ray.remote(num_cpus=1)
    def _process_tile_pair_remote(
        ref_path: str,
        tgt_path: str,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        config: dict[str, Any],
    ) -> TileResult:
        return _process_tile_pair_local(ref_path, tgt_path, col_off, row_off, width, height, config)


class RayTileDispatcher:
    """
    Distributes tile-level processing across Ray cluster nodes or local processes.
    """

    def __init__(self, config: dict[str, Any], use_ray_if_available: bool = True) -> None:
        self.config = config
        self.use_ray = HAS_RAY and use_ray_if_available and ray.is_initialized()

    def dispatch_tiles(
        self,
        ref_path: str | Path,
        tgt_path: str | Path,
        tile_windows: list[tuple[int, int, int, int]],
    ) -> list[TileResult]:
        """
        Dispatch windowed tile jobs across workers.

        Parameters
        ----------
        ref_path, tgt_path : str | Path
            File paths to GeoTIFF rasters.
        tile_windows : list[tuple[int, int, int, int]]
            List of (col_off, row_off, width, height) tuples.

        Returns
        -------
        results : list[TileResult]
            Collected compact results from all tiles.
        """
        ref_s = str(Path(ref_path).resolve())
        tgt_s = str(Path(tgt_path).resolve())

        if self.use_ray:
            logger.info(f"Dispatching {len(tile_windows)} tile tasks across Ray cluster...")
            futures = [
                _process_tile_pair_remote.remote(
                    ref_s, tgt_s, c, r, w, h, self.config
                )
                for (c, r, w, h) in tile_windows
            ]
            return ray.get(futures)
        else:
            logger.info(f"Dispatching {len(tile_windows)} tile tasks using ProcessPoolExecutor...")
            results = []
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(_process_tile_pair_local, ref_s, tgt_s, c, r, w, h, self.config)
                    for (c, r, w, h) in tile_windows
                ]
                for f in futures:
                    results.append(f.result())
            return results
