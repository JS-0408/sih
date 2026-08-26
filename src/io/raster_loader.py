"""
src/io/raster_loader.py
~~~~~~~~~~~~~~~~~~~~~~~
GeoTIFF raster loader with CRS validation and tile generation.
Decouples all I/O from downstream compute modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


class InvalidCRSError(Exception):
    """Raised when raster CRS is absent or cannot be reprojected."""


@dataclass
class Tile:
    """A spatially-aware image tile extracted from a raster."""

    data: np.ndarray          # shape (bands, H, W) or (H, W) for single-band
    col_off: int              # pixel column offset in source raster
    row_off: int              # pixel row offset in source raster
    transform: Affine         # affine transform for this tile's origin
    crs: CRS


@dataclass
class RasterMeta:
    """Lightweight metadata container for an opened raster."""

    path: Path
    crs: CRS
    transform: Affine
    width: int
    height: int
    count: int                # number of bands
    dtype: str
    nodata: float | None = None
    extra: dict = field(default_factory=dict)


class RasterLoader:
    """
    Loads GeoTIFF files with strict CRS validation and optional band selection.

    Parameters
    ----------
    crs_target : str
        Expected CRS as an EPSG string (e.g. ``"EPSG:4326"``).
    bands : list[int] | None
        1-indexed band indices to load. ``None`` loads all bands.
    max_memory_mb : int
        Refuse to load rasters larger than this threshold (MB).
    """

    def __init__(
        self,
        crs_target: str = "EPSG:4326",
        bands: list[int] | None = None,
        max_memory_mb: int = 4096,
    ) -> None:
        self.crs_target = CRS.from_string(crs_target)
        self.bands = bands
        self.max_memory_mb = max_memory_mb

    def load(self, path: str | Path) -> tuple[np.ndarray, RasterMeta]:
        """
        Open and validate a GeoTIFF raster.

        Returns
        -------
        data : np.ndarray
            Pixel array, shape ``(bands, H, W)``.
        meta : RasterMeta
            Metadata including CRS and affine transform.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        InvalidCRSError
            If the raster has no CRS or mismatches ``crs_target``.
        MemoryError
            If the raster exceeds ``max_memory_mb``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            if src.crs is None:
                raise InvalidCRSError(f"Raster has no CRS: {path}")
            if not src.crs == self.crs_target:
                raise InvalidCRSError(
                    f"CRS mismatch — expected {self.crs_target}, got {src.crs}"
                )

            # Memory guard
            bands = self.bands or list(range(1, src.count + 1))
            est_mb = (src.width * src.height * len(bands) * 4) / (1024**2)
            if est_mb > self.max_memory_mb:
                raise MemoryError(
                    f"Raster too large: ~{est_mb:.0f} MB > limit {self.max_memory_mb} MB"
                )

            data = src.read(bands)
            meta = RasterMeta(
                path=path,
                crs=src.crs,
                transform=src.transform,
                width=src.width,
                height=src.height,
                count=len(bands),
                dtype=str(data.dtype),
                nodata=src.nodata,
                extra=dict(src.tags()),
            )

        return data, meta

    def tile_generator(
        self,
        data: np.ndarray,
        meta: RasterMeta,
        tile_size: int = 512,
        overlap_pct: float = 0.2,
    ) -> Iterator[Tile]:
        """
        Yield spatially-aware tiles from a raster array.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(bands, H, W)``.
        meta : RasterMeta
            Source raster metadata (for transform propagation).
        tile_size : int
            Tile width and height in pixels.
        overlap_pct : float
            Fractional overlap between adjacent tiles (0.0–1.0).
        """
        stride = int(tile_size * (1.0 - overlap_pct))
        if stride <= 0:
            raise ValueError("overlap_pct must be < 1.0")

        _, H, W = data.shape
        rows = math.ceil((H - tile_size) / stride) + 1
        cols = math.ceil((W - tile_size) / stride) + 1

        for r in range(rows):
            for c in range(cols):
                row_off = min(r * stride, H - tile_size)
                col_off = min(c * stride, W - tile_size)
                tile_data = data[:, row_off: row_off + tile_size, col_off: col_off + tile_size]
                tile_transform = meta.transform * Affine.translation(col_off, row_off)
                yield Tile(
                    data=tile_data,
                    col_off=col_off,
                    row_off=row_off,
                    transform=tile_transform,
                    crs=meta.crs,
                )
