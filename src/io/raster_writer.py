"""
src/io/raster_writer.py
~~~~~~~~~~~~~~~~~~~~~~~
GeoTIFF exporter for aligned raster outputs.

Accepts a numpy array (shape ``(bands, H, W)``), an updated Affine transform,
and a CRS, then writes a fully georeferenced GeoTIFF that can be loaded
directly into QGIS for visual alignment inspection.

Design notes
------------
* ``dtype`` is auto-detected from the input array; override via ``dtype`` arg.
* LZW compression is applied by default to keep output file sizes small.
* A ``nodata`` value can be optionally embedded so QGIS masks transparent
  edges correctly (useful after RANSAC registration crops).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


# Mapping from numpy dtype to rasterio-compatible dtype strings
_DTYPE_MAP: dict[str, str] = {
    "uint8":   "uint8",
    "uint16":  "uint16",
    "uint32":  "uint32",
    "int16":   "int16",
    "int32":   "int32",
    "float32": "float32",
    "float64": "float64",
}

CompressionType = Literal["lzw", "deflate", "none"]


class RasterWriter:
    """
    Writes georeferenced GeoTIFF files from aligned numpy arrays.

    Parameters
    ----------
    crs : str
        Output CRS as an EPSG string (e.g. ``"EPSG:4326"``).
    compress : CompressionType
        Compression algorithm for the output file.
        ``"lzw"``     — lossless, good general-purpose choice (default).
        ``"deflate"`` — lossless, slightly better ratio, slower.
        ``"none"``    — no compression; largest file, fastest write.
    nodata : float | None
        Value to embed as the NoData sentinel.  ``None`` omits NoData.

    Examples
    --------
    >>> writer = RasterWriter(crs="EPSG:4326")
    >>> writer.write(aligned_array, transform, output_path)
    """

    def __init__(
        self,
        crs: str = "EPSG:4326",
        compress: CompressionType = "lzw",
        nodata: float | None = None,
    ) -> None:
        self.crs = CRS.from_string(crs)
        self.compress = compress
        self.nodata = nodata

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_dtype(self, array: np.ndarray, override: str | None) -> str:
        """Return a rasterio-compatible dtype string."""
        key = override or str(array.dtype)
        if key not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype '{key}'. "
                f"Supported: {list(_DTYPE_MAP.keys())}"
            )
        return _DTYPE_MAP[key]

    def _normalise_array(self, array: np.ndarray) -> np.ndarray:
        """
        Ensure array is 3-D with shape ``(bands, H, W)``.

        Accepts:
        * ``(H, W)``       — single-band grayscale
        * ``(bands, H, W)``— multi-band (rasterio-native)
        """
        if array.ndim == 2:
            return array[np.newaxis, :, :]   # (1, H, W)
        if array.ndim == 3:
            return array
        raise ValueError(
            f"Expected 2-D or 3-D array; got shape {array.shape}."
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def write(
        self,
        array: np.ndarray,
        transform: Affine,
        path: str | Path,
        *,
        dtype: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Write ``array`` as a georeferenced GeoTIFF.

        Parameters
        ----------
        array : np.ndarray
            Pixel data with shape ``(bands, H, W)`` or ``(H, W)``
            (single-band shorthand).
        transform : Affine
            Affine transform for the output raster's origin and resolution.
            After image registration this is typically the updated transform
            computed from the homography matrix.
        path : str | Path
            Destination file path.  Parent directories are created
            automatically.
        dtype : str | None
            Override output dtype (e.g. ``"uint8"``).  Defaults to the
            array's own dtype.
        overwrite : bool
            If ``False`` (default), raises ``FileExistsError`` when the
            destination already exists.

        Returns
        -------
        Path
            Absolute path of the written file.

        Raises
        ------
        FileExistsError
            If ``path`` exists and ``overwrite=False``.
        ValueError
            If the array dimensions or dtype are unsupported.
        """
        path = Path(path).resolve()
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. "
                "Pass overwrite=True to replace it."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        array = self._normalise_array(array)
        bands, height, width = array.shape
        rio_dtype = self._resolve_dtype(array, dtype)

        profile: dict = {
            "driver":    "GTiff",
            "dtype":     rio_dtype,
            "width":     width,
            "height":    height,
            "count":     bands,
            "crs":       self.crs,
            "transform": transform,
            "compress":  self.compress if self.compress != "none" else None,
        }
        if self.nodata is not None:
            profile["nodata"] = self.nodata

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array)

        return path

    def write_tile(
        self,
        array: np.ndarray,
        transform: Affine,
        output_dir: str | Path,
        tile_id: str,
        *,
        dtype: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Convenience wrapper to write a single tile into an output directory.

        The file is named ``tile_<tile_id>.tif``.

        Parameters
        ----------
        array : np.ndarray
            Tile pixel data, shape ``(bands, H, W)`` or ``(H, W)``.
        transform : Affine
            Tile-level affine transform.
        output_dir : str | Path
            Directory to write into (created if absent).
        tile_id : str
            Unique identifier appended to the filename (e.g. ``"r0_c1"``).
        dtype : str | None
            Optional dtype override.
        overwrite : bool
            Replace existing file if ``True``.

        Returns
        -------
        Path
            Path of the written tile file.
        """
        output_dir = Path(output_dir)
        return self.write(
            array=array,
            transform=transform,
            path=output_dir / f"tile_{tile_id}.tif",
            dtype=dtype,
            overwrite=overwrite,
        )
