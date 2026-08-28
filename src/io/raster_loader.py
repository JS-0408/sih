"""
src/io/raster_loader.py
~~~~~~~~~~~~~~~~~~~~~~~
GeoTIFF raster loader with CRS validation and memory-safe windowed tile generation.
Decouples all I/O from downstream compute modules.

Memory strategy
---------------
* ``load()``           — eager-loads small rasters (≤ max_memory_mb) into RAM.
* ``tile_generator()`` — streams tiles directly from GeoTIFF using
                         ``rasterio.windows.Window``; only one tile is ever
                         resident in memory at a time. Calls ``gc.collect()``
                         after each yield so freed memory is returned to the OS
                         before the next block is read. Safe for 8 GB machines.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window


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

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _validate_src(self, src: rasterio.DatasetReader, path: Path) -> None:
        """Raise if CRS is absent or mismatches ``crs_target``."""
        if src.crs is None:
            raise InvalidCRSError(f"Raster has no CRS: {path}")
        if not src.crs == self.crs_target:
            raise InvalidCRSError(
                f"CRS mismatch — expected {self.crs_target}, got {src.crs}"
            )

    def load(self, path: str | Path) -> tuple[np.ndarray, RasterMeta]:
        """
        Open and validate a GeoTIFF raster (eager load into RAM).

        Suitable for small rasters (≤ ``max_memory_mb``).  For large files
        use ``tile_generator(path, ...)`` which streams tiles from disk.

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
            self._validate_src(src, path)

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
        source: Union[str, Path, np.ndarray],
        meta: RasterMeta | None = None,
        tile_size: int = 512,
        overlap_pct: float = 0.2,
    ) -> Iterator[Tile]:
        """
        Yield spatially-aware tiles from a GeoTIFF file **or** a pre-loaded array.

        Memory-safe path (recommended for 8 GB machines)
        -------------------------------------------------
        Pass a file ``path`` as ``source``.  Each tile is read directly from
        disk using ``rasterio.windows.Window`` — the full raster is *never*
        loaded into RAM.  After every ``yield``, ``gc.collect()`` is called
        so Python releases the previous tile's memory before the next block
        is read.

        Legacy / array path (backward-compatible)
        -----------------------------------------
        Pass a ``np.ndarray`` (shape ``(bands, H, W)``) together with a
        ``RasterMeta`` object.  Behaves like the original implementation;
        no windowed I/O is performed.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Either a filesystem path to a GeoTIFF **or** a pre-loaded
            ``(bands, H, W)`` NumPy array.
        meta : RasterMeta | None
            Required when ``source`` is a NumPy array; ignored (re-read
            from file) when ``source`` is a path.
        tile_size : int
            Tile width and height in pixels.
        overlap_pct : float
            Fractional overlap between adjacent tiles (0.0–1.0).

        Yields
        ------
        Tile
            One spatially-aware tile per iteration.

        Raises
        ------
        ValueError
            If ``overlap_pct >= 1.0`` or ``meta`` is missing for array input.
        InvalidCRSError
            If the raster file has no CRS or a CRS mismatch (file path only).
        """
        stride = int(tile_size * (1.0 - overlap_pct))
        if stride <= 0:
            raise ValueError("overlap_pct must be < 1.0")

        # ── Array (legacy) path ──────────────────────────────────────────
        if isinstance(source, np.ndarray):
            if meta is None:
                raise ValueError("meta must be provided when source is a NumPy array.")
            data = source
            _, H, W = data.shape
            rows = math.ceil(max(H - tile_size, 0) / stride) + 1
            cols = math.ceil(max(W - tile_size, 0) / stride) + 1
            for r in range(rows):
                for c in range(cols):
                    row_off = min(r * stride, max(H - tile_size, 0))
                    col_off = min(c * stride, max(W - tile_size, 0))
                    tile_data = data[
                        :,
                        row_off: row_off + tile_size,
                        col_off: col_off + tile_size,
                    ]
                    yield Tile(
                        data=tile_data,
                        col_off=col_off,
                        row_off=row_off,
                        transform=meta.transform * Affine.translation(col_off, row_off),
                        crs=meta.crs,
                    )
            return

        # ── Windowed streaming path (memory-safe, recommended) ───────────
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            self._validate_src(src, path)
            bands = self.bands or list(range(1, src.count + 1))
            H, W = src.height, src.width
            base_transform: Affine = src.transform
            crs: CRS = src.crs

        rows = math.ceil(max(H - tile_size, 0) / stride) + 1
        cols = math.ceil(max(W - tile_size, 0) / stride) + 1

        for r in range(rows):
            for c in range(cols):
                row_off = min(r * stride, max(H - tile_size, 0))
                col_off = min(c * stride, max(W - tile_size, 0))

                # Clamp to raster bounds at edges
                actual_h = min(tile_size, H - row_off)
                actual_w = min(tile_size, W - col_off)

                with rasterio.open(path) as src:
                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=actual_w,
                        height=actual_h,
                    )
                    tile_data = src.read(bands, window=window)

                # Pad to full tile_size if edge tile is smaller
                if actual_h < tile_size or actual_w < tile_size:
                    padded = np.zeros(
                        (len(bands), tile_size, tile_size), dtype=tile_data.dtype
                    )
                    padded[:, :actual_h, :actual_w] = tile_data
                    tile_data = padded

                tile_transform = base_transform * Affine.translation(col_off, row_off)

                yield Tile(
                    data=tile_data,
                    col_off=col_off,
                    row_off=row_off,
                    transform=tile_transform,
                    crs=crs,
                )

                # 8 GB Guard: release tile memory before reading next block
                del tile_data
                gc.collect()
