"""
benchmark/run.py
~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License

Phase 10 Benchmark Runner — Scientific Comparison for ISRO SIH26166.

Evaluates pipeline configurations across multiple simulated Chandrayaan-2 test conditions:
1. Same-sensor baseline (ideal conditions)
2. Solar illumination variation (gamma/contrast shift)
3. Multi-scale GSD jump (resolution downsampling)
4. Viewpoint rotation/affine distortion
5. Cross-sensor synthetic pair

Generates a machine-readable CSV/JSON benchmark report and prints a Markdown summary table.

Usage
-----
python -m benchmark.run
# or
python benchmark/run.py
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run_pipeline, load_config
from src.io.raster_writer import RasterWriter
from rasterio.transform import from_origin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


def create_synthetic_test_pair(
    output_dir: Path,
    scenario_name: str,
    size: int = 1024,
    rotation_deg: float = 3.0,
    scale: float = 1.0,
    gamma: float = 1.0,
    noise_std: float = 0.0,
) -> tuple[Path, Path]:
    """Generate reference & target GeoTIFF rasters for a benchmark scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / f"{scenario_name}_ref.tif"
    tgt_path = output_dir / f"{scenario_name}_tgt.tif"

    # Generate synthetic crater surface
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    xx, yy = np.meshgrid(x, y)
    surface = np.zeros((size, size), dtype=np.float32)

    # Add 25 synthetic craters
    for _ in range(25):
        cx, cy = rng.uniform(-2.5, 2.5, 2)
        r = rng.uniform(0.1, 0.4)
        depth = rng.uniform(30, 90)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        crater = depth * np.exp(-(dist**2) / (2 * r**2))
        surface += crater

    # Normalise base surface to uint8
    lo, hi = surface.min(), surface.max()
    ref_arr = ((surface - lo) / max(hi - lo, 1) * 255).astype(np.uint8)

    # Create target with geometric transform (rotation + scale + translation)
    h, w = ref_arr.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle=rotation_deg, scale=scale)
    rot_mat[0, 2] += 18.0   # +18px X translation
    rot_mat[1, 2] += -12.0  # -12px Y translation

    tgt_arr = cv2.warpAffine(ref_arr, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

    # Apply solar illumination shift (gamma correction)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
        tgt_arr = cv2.LUT(tgt_arr, table)

    # Add Gaussian noise
    if noise_std > 0:
        noise = rng.normal(0, noise_std, tgt_arr.shape).astype(np.float32)
        tgt_arr = np.clip(tgt_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Write GeoTIFF rasters
    transform = from_origin(77.5, 12.9, 0.0001, 0.0001)
    writer = RasterWriter(crs="EPSG:4326", compress="lzw")
    writer.write(ref_arr, transform, ref_path, overwrite=True)
    writer.write(tgt_arr, transform, tgt_path, overwrite=True)

    return ref_path, tgt_path


def run_benchmark() -> None:
    """Run full benchmark suite and export results."""
    bench_dir = Path("benchmark")
    data_dir = bench_dir / "data"
    results_dir = bench_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"name": "1_same_sensor_baseline", "rot": 2.0, "scale": 1.0, "gamma": 1.0, "noise": 0.0},
        {"name": "2_solar_illumination",   "rot": 2.0, "scale": 1.0, "gamma": 2.2, "noise": 5.0},
        {"name": "3_viewpoint_affine",     "rot": 6.5, "scale": 1.03, "gamma": 1.0, "noise": 0.0},
        {"name": "4_multimodal_extreme",   "rot": 4.0, "scale": 1.02, "gamma": 2.5, "noise": 10.0},
    ]

    strategies = [
        {"id": "ORB_baseline",            "detector": "ORB",  "prep": "raw",      "subpixel": False},
        {"id": "SIFT_baseline",           "detector": "SIFT", "prep": "raw",      "subpixel": False},
        {"id": "SIFT_CLAHE",              "detector": "SIFT", "prep": "clahe",    "subpixel": False},
        {"id": "SIFT_Gradient",           "detector": "SIFT", "prep": "gradient", "subpixel": False},
        {"id": "SIFT_CLAHE_SubPixel",     "detector": "SIFT", "prep": "clahe",    "subpixel": True},
    ]

    base_config = load_config("config/pipeline_config.yaml")

    rows = []

    print("\n" + "=" * 90)
    print("  YAAZHI GEOALIGN OS — BENCHMARK SUITE (ISRO SIH26166)")
    print("=" * 90)

    for sc in scenarios:
        sc_name = sc["name"]
        print(f"\n[Scenario: {sc_name}] (Rotation={sc['rot']}°, Gamma={sc['gamma']}, Noise={sc['noise']})")
        ref_path, tgt_path = create_synthetic_test_pair(
            data_dir, sc_name,
            rotation_deg=sc["rot"], scale=sc["scale"],
            gamma=sc["gamma"], noise_std=sc["noise"],
        )

        for strat in strategies:
            strat_id = strat["id"]
            out_path = data_dir / f"{sc_name}_{strat_id}_out.tif"

            # Customise config for strategy
            cfg = json.loads(json.dumps(base_config))  # deep copy
            cfg["keypoints"]["method"] = strat["detector"]
            cfg["preprocessing"]["mode"] = strat["prep"]
            cfg["refinement"]["enabled"] = strat["subpixel"]

            try:
                summary = run_pipeline(ref_path, tgt_path, out_path, cfg)
                status = summary["status"]
                inliers = summary["features"]["gcp_inliers"]
                fit_rmse = summary["metrics"]["fitting_rmse_px"] or 999.0
                val_report = summary.get("independent_validation", {})
                val_p95 = val_report.get("p95_px", 999.0)
                val_rmse = val_report.get("rmse_px", 999.0)
                coverage = summary["metrics"]["spatial_coverage"]
                runtime = summary["runtime_seconds"]

                row = {
                    "scenario": sc_name,
                    "strategy": strat_id,
                    "detector": strat["detector"],
                    "prep": strat["prep"],
                    "subpixel": strat["subpixel"],
                    "status": status,
                    "inliers": inliers,
                    "fit_rmse_px": fit_rmse,
                    "val_rmse_px": val_rmse,
                    "val_p95_px": val_p95,
                    "coverage_pct": round(coverage * 100, 1),
                    "runtime_sec": runtime,
                }
            except Exception as exc:
                row = {
                    "scenario": sc_name,
                    "strategy": strat_id,
                    "detector": strat["detector"],
                    "prep": strat["prep"],
                    "subpixel": strat["subpixel"],
                    "status": f"ERROR ({exc})",
                    "inliers": 0,
                    "fit_rmse_px": 999.0,
                    "val_rmse_px": 999.0,
                    "val_p95_px": 999.0,
                    "coverage_pct": 0.0,
                    "runtime_sec": 0.0,
                }

            rows.append(row)
            sub_flag = "SubPixel" if strat["subpixel"] else "Coarse"
            print(
                f"  |-- {strat_id:<22} | Status: {row['status']:<24} | "
                f"Inliers: {row['inliers']:<4} | Val RMSE: {row['val_rmse_px']:<6.3f} px | "
                f"P95: {row['val_p95_px']:<6.3f} px | Time: {row['runtime_sec']}s"
            )

    # Export CSV report
    csv_path = results_dir / "benchmark_report.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Export JSON report
    json_path = results_dir / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark_results": rows}, f, indent=2)

    print("\n" + "=" * 90)
    print(f"[OK] Benchmark Complete. Reports exported to:\n  - CSV : {csv_path}\n  - JSON: {json_path}")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
