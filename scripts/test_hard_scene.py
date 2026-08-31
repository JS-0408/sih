"""
scripts/test_hard_scene.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates a challenging multi-band, large-resolution (2048x2048) GeoTIFF input dataset
with scale factor, rotation angle (5 degrees), translation shift, and texture variation
to test the registration pipeline across diverse inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run_pipeline, load_config


def generate_hard_dataset(
    output_dir: str | Path = "data/hard_scene",
    width: int = 2048,
    height: int = 2048,
    scale: float = 1.05,
    angle_deg: float = 4.5,
    shift_x: float = 65.0,
    shift_y: float = -35.0,
) -> tuple[Path, Path]:
    """Generate 3-band RGB GeoTIFFs with scale, rotation, translation, and noise."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(101)

    # Base texture channels (3 bands: Red, Green, Blue)
    b1 = np.random.normal(120, 25, (height, width))
    b2 = np.random.normal(100, 20, (height, width))
    b3 = np.random.normal(80, 15, (height, width))

    # Add craters and terrain features
    for _ in range(60):
        cx = np.random.randint(100, width - 100)
        cy = np.random.randint(100, height - 100)
        r = np.random.randint(20, 60)

        cv2.circle(b1, (cx, cy), r, (240,), 4)
        cv2.circle(b2, (cx, cy), r, (210,), 4)
        cv2.circle(b3, (cx, cy), r, (180,), 4)

        cv2.circle(b1, (cx + 3, cy + 3), r - 5, (30,), -1)
        cv2.circle(b2, (cx + 3, cy + 3), r - 5, (40,), -1)
        cv2.circle(b3, (cx + 3, cy + 3), r - 5, (50,), -1)

    ref_img = np.stack([
        np.clip(b1, 0, 255).astype(np.uint8),
        np.clip(b2, 0, 255).astype(np.uint8),
        np.clip(b3, 0, 255).astype(np.uint8),
    ], axis=0)  # (3, 2048, 2048)

    # Transform target image (scale + rotation + shift)
    center = (width / 2.0, height / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, scale=scale)
    rot_mat[0, 2] += shift_x
    rot_mat[1, 2] += shift_y

    tgt_bands = []
    for b in range(3):
        warped_b = cv2.warpAffine(
            ref_img[b], rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderValue=100
        )
        tgt_bands.append(warped_b)
    tgt_img = np.stack(tgt_bands, axis=0)

    # Geospatial transform (EPSG:32643 - WGS 84 / UTM Zone 43N meters)
    ref_transform = from_origin(300000.0, 1400000.0, 10.0, 10.0)

    ref_path = out_dir / "reference_2k.tif"
    tgt_path = out_dir / "target_2k.tif"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": ref_transform,
        "compress": "lzw",
    }

    with rasterio.open(ref_path, "w", **profile) as dst:
        dst.write(ref_img)

    with rasterio.open(tgt_path, "w", **profile) as dst:
        dst.write(tgt_img)

    print(f"[OK] Generated 3-Band 2048x2048 Reference: {ref_path}")
    print(f"[OK] Generated 3-Band 2048x2048 Target:    {tgt_path}")
    return ref_path, tgt_path


def run_hard_test() -> None:
    ref_path, tgt_path = generate_hard_dataset()
    out_path = Path("outputs/registered_2k_hard.tif")

    config = load_config("config/phase1_config.yaml")
    
    # Customize for 2K scene
    config["tiling"]["tile_size"] = 512
    config["tiling"]["overlap_pct"] = 0.2
    config["geospatial"]["crs_target"] = "EPSG:32643"
    config["keypoints"]["max_keypoints"] = 3000
    config.setdefault("evaluation", {})["min_inliers"] = 10
    config["evaluation"]["max_rmse"] = 10.0

    print("\nExecuting registration pipeline on 2048x2048 3-band input...")
    summary = run_pipeline(
        reference_path=ref_path,
        target_path=tgt_path,
        output_path=out_path,
        config=config,
    )

    print("\n" + "=" * 60)
    print("  TEST RUN RESULTS (2048x2048 Multi-Band Input)")
    print("=" * 60)
    print(f"  Status           : {summary['status']}")
    print(f"  Runtime          : {summary['runtime_seconds']}s")
    print(f"  Tiles Processed  : {summary['tiling']['grid']} ({summary['tiling']['total_tiles']} tiles)")
    print(f"  GCPs Collected   : {summary['features']['total_gcps_collected']}")
    print(f"  GCP Inliers      : {summary['features']['gcp_inliers']}")
    print(f"  Global RMSE (px) : {summary['metrics']['fitting_rmse_px']} px")
    print(f"  Spatial Coverage : {summary['metrics']['spatial_coverage'] * 100:.1f}%")
    print(f"  Output GeoTIFF   : {summary['files']['output']}")
    print("=" * 60)


if __name__ == "__main__":
    run_hard_test()
