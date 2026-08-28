"""
scripts/generate_synthetic_geotiff.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates synthetic pair of reference and target GeoTIFF images with known affine transform
and terrain features (craters, grid lines, high-contrast textures) for automated pipeline testing.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin


def generate_synthetic_scene(
    width: int = 1024,
    height: int = 1024,
    num_craters: int = 40,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 2D synthetic terrain raster with craters and distinctive keypoint features."""
    np.random.seed(seed)
    img = np.full((height, width), 128, dtype=np.uint8)

    # Add background Gaussian noise texture
    noise = np.random.normal(0, 15, (height, width))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Add craters (concentric dark/light circles)
    for _ in range(num_craters):
        cx = np.random.randint(50, width - 50)
        cy = np.random.randint(50, height - 50)
        r = np.random.randint(15, 45)

        # Outer rim
        cv2.circle(img, (cx, cy), r, (220,), 3)
        # Inner shadow
        cv2.circle(img, (cx + 2, cy + 2), r - 4, (40,), -1)
        # Floor
        cv2.circle(img, (cx + 4, cy + 4), r - 8, (100,), -1)

    # Add geometric grid lines / feature landmarks
    for i in range(100, width, 200):
        cv2.line(img, (i, 0), (i, height), (180,), 2)
        cv2.line(img, (0, i), (width, i), (180,), 2)

    return img


def create_synthetic_geotiffs(
    output_dir: str | Path = "data",
    width: int = 1024,
    height: int = 1024,
    shift_x: float = 25.0,
    shift_y: float = -15.0,
    angle_deg: float = 2.5,
) -> tuple[Path, Path]:
    """
    Create reference and transformed target GeoTIFF files.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save synthetic GeoTIFF files.
    width, height : int
        Raster dimensions in pixels.
    shift_x, shift_y : float
        Known translation offset in pixels.
    angle_deg : float
        Known rotation angle in degrees.

    Returns
    -------
    ref_path, tgt_path : tuple[Path, Path]
        Paths to created reference and target GeoTIFF files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_img = generate_synthetic_scene(width=width, height=height)

    # Create target image with known Affine transformation (shift + rotation)
    center = (width / 2.0, height / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
    rot_mat[0, 2] += shift_x
    rot_mat[1, 2] += shift_y

    tgt_img = cv2.warpAffine(
        ref_img, rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderValue=128
    )

    # Define Geospatial Transform (EPSG:4326 origin at Lat 12.0, Lon 77.0, 0.0001 deg/px)
    ref_transform = from_origin(77.0, 12.0, 0.0001, 0.0001)

    ref_path = out_dir / "reference.tif"
    tgt_path = out_dir / "target.tif"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": ref_transform,
        "compress": "lzw",
    }

    with rasterio.open(ref_path, "w", **profile) as dst:
        dst.write(ref_img, 1)

    with rasterio.open(tgt_path, "w", **profile) as dst:
        dst.write(tgt_img, 1)

    print(f"[OK] Synthetic reference GeoTIFF created at: {ref_path}")
    print(f"[OK] Synthetic target GeoTIFF created at: {tgt_path}")
    return ref_path, tgt_path


if __name__ == "__main__":
    create_synthetic_geotiffs()
