"""
main.py
~~~~~~~
Copyright (c) 2026 Santhosh — MIT License

Production CLI & Orchestration Engine — Geospatial Image Registration Pipeline.
Part of the Yaazhi GeoAlign OS / ISRO SIH26166 Chandrayaan-2 Registration System.

Pipeline Workflow
-----------------
1. Preflight  : Validate config, paths, CRS compatibility.
2. Tile Plan  : Generate deterministic windowed tile grid.
3. Per-tile   : SIFT detection → grid pruning → FLANN → RANSAC → collect GCPs.
4. GCP Global : Fit one global homography/affine from all tile inliers.
5. Warp       : Resample target raster with the global transform and export GeoTIFF.
6. Evaluate   : Quality gate check, RMSE, spatial coverage, summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.geometry.gcp_estimator import GCPEstimator, ControlPoint
from src.io.raster_loader import RasterLoader
from src.io.raster_writer import RasterWriter
from src.matching.flann_matcher import FLANNMatcher
from src.matching.ransac_filter import RANSACFilter
from src.metrics.evaluator import PipelineEvaluator, TileStats
from src.metrics.rmse_calculator import RMSECalculator
from src.processing.grid_filter import GridFilter
from src.processing.keypoint_detector import KeypointDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PipelineOrchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config: {path}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Single-tile feature extraction + matching
# ─────────────────────────────────────────────────────────────────────────────

def process_tile_pair(
    ref_tile: np.ndarray,
    tgt_tile: np.ndarray,
    tile_id: str,
    detector: KeypointDetector,
    grid_filter: GridFilter,
    matcher: FLANNMatcher,
    ransac: RANSACFilter,
) -> tuple[TileStats, list[ControlPoint]]:
    """Run feature pipeline on one tile pair; return stats and control points."""

    kp_ref, desc_ref = detector.detect(ref_tile)
    kp_tgt, desc_tgt = detector.detect(tgt_tile)

    h, w = ref_tile.shape[-2], ref_tile.shape[-1]
    kp_ref_f, desc_ref_f = grid_filter.filter(kp_ref, (h, w), desc_ref)
    kp_tgt_f, desc_tgt_f = grid_filter.filter(kp_tgt, (h, w), desc_tgt)

    empty_stats = TileStats(
        tile_id=tile_id, raw_matches=0, inlier_count=0,
        inlier_ratio=0.0, rmse_px=None, status="failed",
    )

    if len(kp_ref_f) < 4 or len(kp_tgt_f) < 4 or desc_ref_f is None or desc_tgt_f is None:
        return empty_stats, []

    try:
        raw_matches = matcher.match(desc_ref_f, desc_tgt_f)
    except ValueError:
        return empty_stats, []

    if len(raw_matches) < 4:
        empty_stats.raw_matches = len(raw_matches)
        return empty_stats, []

    # Homography: target → reference direction
    inv_matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in raw_matches]
    try:
        inliers, H = ransac.filter(kp_tgt_f, kp_ref_f, inv_matches)
    except ValueError:
        empty_stats.raw_matches = len(raw_matches)
        return empty_stats, []

    if not inliers or H is None:
        empty_stats.raw_matches = len(raw_matches)
        return empty_stats, []

    # Collect control points
    src_pts = np.float32([kp_tgt_f[m.queryIdx].pt for m in inliers])
    dst_pts = np.float32([kp_ref_f[m.trainIdx].pt for m in inliers])

    gcps = GCPEstimator().add_tile_correspondences(src_pts, dst_pts, tile_id=tile_id)

    inlier_ratio = len(inliers) / len(raw_matches)
    stats = TileStats(
        tile_id=tile_id,
        raw_matches=len(raw_matches),
        inlier_count=len(inliers),
        inlier_ratio=round(inlier_ratio, 4),
        rmse_px=None,
        status="ok",
    )
    return stats, gcps


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    reference_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """End-to-end multi-tile geospatial image registration pipeline."""

    t0 = time.time()
    ref_p = Path(reference_path).resolve()
    tgt_p = Path(target_path).resolve()
    out_p = Path(output_path).resolve()

    for p in (ref_p, tgt_p):
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {p}")

    cfg_kp     = config.get("keypoints", {})
    cfg_tiling = config.get("tiling", {})
    cfg_flann  = config.get("flann", {})
    cfg_ransac = config.get("ransac", {})
    cfg_geo    = config.get("geospatial", {})
    cfg_out    = config.get("metrics", {})
    cfg_eval   = config.get("evaluation", {})

    crs_target = cfg_geo.get("crs_target", "EPSG:4326")
    bands      = cfg_geo.get("bands", None)
    tile_size  = cfg_tiling.get("tile_size", 512)
    overlap    = cfg_tiling.get("overlap_pct", 0.1)

    # ── Preflight: load reference fully (used for warp target size) ──────
    logger.info("Preflight: loading rasters…")
    loader = RasterLoader(crs_target=crs_target, bands=bands)
    ref_data, ref_meta = loader.load(ref_p)
    tgt_data, _       = loader.load(tgt_p)

    ref_h = ref_data.shape[1] if ref_data.ndim == 3 else ref_data.shape[0]
    ref_w = ref_data.shape[2] if ref_data.ndim == 3 else ref_data.shape[1]

    logger.info(f"Reference: {ref_data.shape}  CRS: {ref_meta.crs}")

    # ── Build tile grid ───────────────────────────────────────────────────
    stride = int(tile_size * (1.0 - overlap))
    rows_n = max(1, (ref_h - tile_size + stride) // stride + 1)
    cols_n = max(1, (ref_w - tile_size + stride) // stride + 1)
    logger.info(f"Tile grid: {rows_n}×{cols_n} ({rows_n * cols_n} tiles, size={tile_size}, overlap={overlap})")

    # ── Initialise pipeline components ───────────────────────────────────
    detector    = KeypointDetector(method=cfg_kp.get("method", "SIFT"),
                                   max_keypoints=cfg_kp.get("max_keypoints", 2000))
    grid_filter = GridFilter(grid_cells=cfg_kp.get("grid_cells", 8))
    matcher     = FLANNMatcher(trees=cfg_flann.get("trees", 5),
                               checks=cfg_flann.get("checks", 50),
                               ratio_threshold=cfg_flann.get("ratio_threshold", 0.75))
    ransac      = RANSACFilter(threshold=cfg_ransac.get("threshold", 5.0),
                               max_iter=cfg_ransac.get("max_iter", 2000),
                               confidence=cfg_ransac.get("confidence", 0.995))

    # ── Per-tile processing ───────────────────────────────────────────────
    all_gcps: list[ControlPoint] = []
    tile_stats_list: list[TileStats] = []

    for ri in range(rows_n):
        for ci in range(cols_n):
            r_off = min(ri * stride, max(ref_h - tile_size, 0))
            c_off = min(ci * stride, max(ref_w - tile_size, 0))
            act_h = min(tile_size, ref_h - r_off)
            act_w = min(tile_size, ref_w - c_off)

            tile_id = f"r{ri}_c{ci}"

            if ref_data.ndim == 3:
                ref_tile = ref_data[:, r_off:r_off+act_h, c_off:c_off+act_w]
                tgt_tile = tgt_data[:, r_off:r_off+act_h, c_off:c_off+act_w]
            else:
                ref_tile = ref_data[r_off:r_off+act_h, c_off:c_off+act_w]
                tgt_tile = tgt_data[r_off:r_off+act_h, c_off:c_off+act_w]

            stats, gcps = process_tile_pair(
                ref_tile, tgt_tile, tile_id,
                detector, grid_filter, matcher, ransac,
            )

            # Shift tile-local GCP coords to global raster coordinates
            for gcp in gcps:
                gcp.src_x += c_off
                gcp.src_y += r_off
                gcp.dst_x += c_off
                gcp.dst_y += r_off

            tile_stats_list.append(stats)
            all_gcps.extend(gcps)

    logger.info(f"Total GCPs collected: {len(all_gcps)}")

    # ── Global GCP estimation ─────────────────────────────────────────────
    gcp_model = cfg_ransac.get("model", "homography")
    estimator = GCPEstimator(
        model=gcp_model,
        min_inliers=cfg_eval.get("min_inliers", 4),
        max_rmse_px=cfg_eval.get("max_rmse", 15.0),
        ransac_threshold=cfg_ransac.get("threshold", 5.0),
        min_coverage=cfg_eval.get("min_coverage", 0.05),
    )
    gcp_result = estimator.estimate(all_gcps, ref_w, ref_h)

    homography = gcp_result.matrix

    # ── Warp & export ─────────────────────────────────────────────────────
    logger.info("Resampling target raster with global homography…")
    if tgt_data.ndim == 3:
        bands_n, _, _ = tgt_data.shape
        warped = np.stack([
            cv2.warpPerspective(tgt_data[b], homography, (ref_w, ref_h), flags=cv2.INTER_LINEAR)
            for b in range(bands_n)
        ], axis=0)
    else:
        warped = cv2.warpPerspective(tgt_data, homography, (ref_w, ref_h), flags=cv2.INTER_LINEAR)

    writer      = RasterWriter(crs=str(ref_meta.crs),
                               compress=cfg_out.get("compress", "lzw"),
                               nodata=ref_meta.nodata)
    output_file = writer.write(warped, ref_meta.transform, out_p, overwrite=True)

    # ── Evaluate ──────────────────────────────────────────────────────────
    residuals = np.array([cp.residual_px for cp in gcp_result.control_points
                          if cp.residual_px > 0]) if gcp_result.control_points else None

    evaluator = PipelineEvaluator(
        min_inliers=cfg_eval.get("min_inliers", 4),
        max_rmse_px=cfg_eval.get("max_rmse", 15.0),
        min_coverage=cfg_eval.get("min_coverage", 0.05),
        min_tile_success_rate=cfg_eval.get("min_tile_success_rate", 0.3),
    )
    report = evaluator.evaluate(
        tile_stats=tile_stats_list,
        global_rmse_px=gcp_result.rmse_px if not math.isinf(gcp_result.rmse_px) else 999.0,
        spatial_coverage=gcp_result.spatial_coverage,
        inlier_residuals=residuals,
    )

    elapsed = time.time() - t0

    summary = {
        "status": "SUCCESS" if report.overall_pass else "QUALITY_GATE_FAILED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed, 3),
        "files": {
            "reference": str(ref_p),
            "target": str(tgt_p),
            "output": str(output_file),
        },
        "tiling": {
            "tile_size": tile_size,
            "overlap": overlap,
            "grid": f"{rows_n}x{cols_n}",
            "total_tiles": rows_n * cols_n,
        },
        "features": {
            "detector": cfg_kp.get("method", "SIFT"),
            "total_gcps_collected": len(all_gcps),
            "gcp_inliers": gcp_result.inlier_count,
            "gcp_model": gcp_model,
        },
        "metrics": {
            "global_rmse_px": round(gcp_result.rmse_px, 4) if not math.isinf(gcp_result.rmse_px) else None,
            "spatial_coverage": round(gcp_result.spatial_coverage, 4),
            "global_inlier_ratio": report.global_inlier_ratio,
        },
        "evaluation": report.to_dict(),
        "homography_matrix": homography.tolist(),
        "warnings": gcp_result.warnings + report.warnings,
    }

    summary_path = out_p.parent / f"{out_p.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done in {elapsed:.2f}s — Status: {summary['status']} — Summary: {summary_path}")
    return summary





# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geospatial Image Registration Pipeline CLI"
    )
    parser.add_argument("--config",    default="config/pipeline_config.yaml")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--target",    required=True)
    parser.add_argument("--output",    default="outputs/registered_output.tif")
    args = parser.parse_args()

    try:
        cfg     = load_config(args.config)
        summary = run_pipeline(args.reference, args.target, args.output, cfg)
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary["evaluation"]["overall_pass"] else 1)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
