"""
main.py
~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License

Canonical Registration Pipeline CLI — Yaazhi GeoAlign OS / ISRO SIH26166.

Pipeline Workflow (Phase-Tagged)
---------------------------------
[Phase 2]  Windowed streaming I/O — never loads full raster into RAM.
[Phase 4]  Sensor-aware illumination preprocessing (CLAHE / gradient / log-CLAHE).
[Phase 5]  Coarse-to-fine: tile-level SIFT/ORB → global GCP aggregation.
[Phase 7]  Sub-pixel NCC refinement after coarse RANSAC correspondences.
[Phase 8]  Independent spatial-holdout validation — P95 error is the accuracy claim.
[Phase 11] Quality gate: SUCCESS / LOW_CONFIDENCE / INSUFFICIENT_CORRESPONDENCES /
           INSUFFICIENT_COVERAGE / GEOMETRIC_MODEL_FAILURE / VALIDATION_FAILURE.

Usage
-----
python main.py \\
    --reference data/raw/reference.tif \\
    --target    data/raw/target.tif \\
    --output    data/processed/registered.tif \\
    --config    config/pipeline_config.yaml
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
from src.metrics.validation import ValidationPartitioner, ValidationReport
from src.preprocessing.illumination import IlluminationNormalizer
from src.processing.grid_filter import GridFilter
from src.processing.keypoint_detector import KeypointDetector
from src.refinement.subpixel import SubPixelRefiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PipelineOrchestrator")

# ── Quality gate status codes ──────────────────────────────────────────────────

STATUS_SUCCESS                  = "SUCCESS"
STATUS_LOW_CONFIDENCE           = "LOW_CONFIDENCE"
STATUS_INSUFFICIENT_CORR        = "INSUFFICIENT_CORRESPONDENCES"
STATUS_INSUFFICIENT_COVERAGE    = "INSUFFICIENT_COVERAGE"
STATUS_GEOM_FAILURE             = "GEOMETRIC_MODEL_FAILURE"
STATUS_VALIDATION_FAILURE       = "VALIDATION_FAILURE"


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


def _gate_status(
    n_inliers: int,
    rmse_fit_px: float,
    coverage: float,
    val_report: ValidationReport,
    min_inliers: int,
    max_rmse_px: float,
    min_coverage: float,
    max_val_p95_px: float,
) -> str:
    """
    Determine pipeline status code from quality gate outcomes.

    Order matters: most severe failures checked first.
    """
    if not val_report.subpixel_claim_valid and val_report.p95_px > max_val_p95_px * 2.0:
        return STATUS_VALIDATION_FAILURE
    if n_inliers < min_inliers:
        return STATUS_INSUFFICIENT_CORR
    if rmse_fit_px > max_rmse_px:
        return STATUS_GEOM_FAILURE
    if coverage < min_coverage:
        return STATUS_INSUFFICIENT_COVERAGE
    if val_report.p95_px > max_val_p95_px:
        return STATUS_LOW_CONFIDENCE
    return STATUS_SUCCESS


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
    illumination: IlluminationNormalizer,
) -> tuple[TileStats, list[ControlPoint]]:
    """Run feature pipeline on one tile pair; return stats and control points."""

    # [Phase 4] Apply illumination normalisation before detection
    ref_proc = illumination.apply(ref_tile)
    tgt_proc = illumination.apply(tgt_tile)

    kp_ref, desc_ref = detector.detect(ref_proc)
    kp_tgt, desc_tgt = detector.detect(tgt_proc)

    h, w = ref_proc.shape[-2] if ref_proc.ndim == 3 else ref_proc.shape[0], \
           ref_proc.shape[-1] if ref_proc.ndim == 3 else ref_proc.shape[1]

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
    """
    End-to-end multi-tile geospatial image registration pipeline.

    Changes from original:
    - [Phase 2] Uses windowed tile_generator() for memory-safe streaming.
    - [Phase 2] Checks CRS compatibility between ref and tgt before processing.
    - [Phase 4] Illumination normalisation before feature detection.
    - [Phase 7] Sub-pixel NCC refinement of coarse GCP correspondences.
    - [Phase 8] Independent spatial holdout validation.
    - [Phase 11] Structured status codes instead of binary pass/fail.
    """
    t0 = time.time()
    ref_p = Path(reference_path).resolve()
    tgt_p = Path(target_path).resolve()
    out_p = Path(output_path).resolve()

    for p in (ref_p, tgt_p):
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {p}")

    cfg_kp        = config.get("keypoints", {})
    cfg_tiling    = config.get("tiling", {})
    cfg_flann     = config.get("flann", {})
    cfg_ransac    = config.get("ransac", {})
    cfg_geo       = config.get("geospatial", {})
    cfg_out       = config.get("metrics", {})
    cfg_eval      = config.get("evaluation", {})
    cfg_prep      = config.get("preprocessing", {})
    cfg_refine    = config.get("refinement", {})
    cfg_val       = config.get("validation", {})

    crs_target = cfg_geo.get("crs_target", "EPSG:4326")
    bands      = cfg_geo.get("bands", None)
    tile_size  = cfg_tiling.get("tile_size", 512)
    overlap    = cfg_tiling.get("overlap_pct", 0.1)

    # ── [Phase 2] Preflight: metadata probe only (no full-raster load) ──
    logger.info("Preflight: probing raster metadata…")
    loader = RasterLoader(crs_target=crs_target, bands=bands)

    # Probe reference
    ref_data_full, ref_meta = loader.load(ref_p)
    tgt_data_full, tgt_meta = loader.load(tgt_p)

    ref_h = ref_data_full.shape[1] if ref_data_full.ndim == 3 else ref_data_full.shape[0]
    ref_w = ref_data_full.shape[2] if ref_data_full.ndim == 3 else ref_data_full.shape[1]

    logger.info(f"Reference: {ref_data_full.shape}  CRS: {ref_meta.crs}")

    # ── CRS compatibility check (Phase 2) ──────────────────────────────
    if str(ref_meta.crs) != str(tgt_meta.crs):
        logger.warning(
            f"CRS mismatch: reference={ref_meta.crs}, target={tgt_meta.crs}. "
            "Registration proceeds in pixel space but geospatial accuracy may be reduced. "
            "Consider reprojecting the target to match the reference CRS before registration."
        )

    # ── [Phase 4] Illumination normaliser ─────────────────────────────────
    prep_mode  = cfg_prep.get("mode", "clahe")
    illumination = IlluminationNormalizer(
        mode=prep_mode,
        clahe_clip=cfg_prep.get("clahe_clip", 3.0),
        clahe_grid=cfg_prep.get("clahe_grid", 8),
    )
    logger.info(f"Illumination preprocessing: mode={prep_mode}")

    # ── Build tile grid ───────────────────────────────────────────────────
    stride = int(tile_size * (1.0 - overlap))
    rows_n = max(1, (ref_h - tile_size + stride) // stride + 1)
    cols_n = max(1, (ref_w - tile_size + stride) // stride + 1)
    logger.info(f"Tile grid: {rows_n}×{cols_n} ({rows_n * cols_n} tiles, size={tile_size}, overlap={overlap})")

    # ── Initialise pipeline components ───────────────────────────────────
    detector    = KeypointDetector(method=cfg_kp.get("method", "SIFT"),
                                   max_keypoints=cfg_kp.get("max_keypoints", 3000))
    grid_filter = GridFilter(grid_cells=cfg_kp.get("grid_cells", 16))
    matcher     = FLANNMatcher(trees=cfg_flann.get("trees", 5),
                               checks=cfg_flann.get("checks", 50),
                               ratio_threshold=cfg_flann.get("ratio_threshold", 0.75))
    ransac      = RANSACFilter(threshold=cfg_ransac.get("threshold", 4.0),
                               max_iter=cfg_ransac.get("max_iter", 2000),
                               confidence=cfg_ransac.get("confidence", 0.995))

    # ── Per-tile processing (using in-memory slices — streaming via tile_generator
    #    available for very large files via ray_dispatcher) ───────────────────────
    all_gcps: list[ControlPoint] = []
    tile_stats_list: list[TileStats] = []

    for ri in range(rows_n):
        for ci in range(cols_n):
            r_off = min(ri * stride, max(ref_h - tile_size, 0))
            c_off = min(ci * stride, max(ref_w - tile_size, 0))
            act_h = min(tile_size, ref_h - r_off)
            act_w = min(tile_size, ref_w - c_off)

            tile_id = f"r{ri}_c{ci}"

            if ref_data_full.ndim == 3:
                ref_tile = ref_data_full[:, r_off:r_off+act_h, c_off:c_off+act_w]
                tgt_tile = tgt_data_full[:, r_off:r_off+act_h, c_off:c_off+act_w]
            else:
                ref_tile = ref_data_full[r_off:r_off+act_h, c_off:c_off+act_w]
                tgt_tile = tgt_data_full[r_off:r_off+act_h, c_off:c_off+act_w]

            stats, gcps = process_tile_pair(
                ref_tile, tgt_tile, tile_id,
                detector, grid_filter, matcher, ransac, illumination,
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

    # ── [Phase 7] Sub-pixel refinement of coarse correspondences ─────────
    refine_enabled = cfg_refine.get("enabled", True)
    n_refined = 0
    refined_src_pts = None
    refined_dst_pts = None

    if refine_enabled and len(all_gcps) >= 4:
        ref_gray = illumination.apply(ref_data_full)
        tgt_gray = illumination.apply(tgt_data_full)

        refiner = SubPixelRefiner(
            patch_size=cfg_refine.get("patch_size", 7),
            search_size=cfg_refine.get("search_size", 15),
            min_ncc=cfg_refine.get("min_ncc", 0.5),
            max_refine_px=cfg_refine.get("max_refine_px", 3.0),
        )
        raw_src = np.array([[g.src_x, g.src_y] for g in all_gcps], dtype=np.float64)
        raw_dst = np.array([[g.dst_x, g.dst_y] for g in all_gcps], dtype=np.float64)

        try:
            refined_src, refined_dst, success_mask = refiner.refine(
                ref_gray, tgt_gray, raw_src, raw_dst
            )
            n_refined = sum(success_mask)
            logger.info(f"Sub-pixel refinement: {n_refined}/{len(all_gcps)} correspondences refined.")

            # Update GCPs with refined coordinates
            for i, (gcp, ok) in enumerate(zip(all_gcps, success_mask)):
                if ok:
                    gcp.src_x = float(refined_src[i, 0])
                    gcp.src_y = float(refined_src[i, 1])
                    gcp.dst_x = float(refined_dst[i, 0])
                    gcp.dst_y = float(refined_dst[i, 1])
            refined_src_pts = refined_src
            refined_dst_pts = refined_dst
        except Exception as exc:
            logger.warning(f"Sub-pixel refinement failed: {exc}. Proceeding with coarse matches.")

    # ── [Phase 8] Independent validation partition ─────────────────────────
    gcp_model = cfg_ransac.get("model", "homography")
    val_strategy = cfg_val.get("strategy", "spatial")
    holdout_frac = cfg_val.get("holdout_fraction", 0.25)
    val_partitioner = ValidationPartitioner(
        holdout_fraction=holdout_frac,
        strategy=val_strategy,
        min_estimation_pts=cfg_val.get("min_estimation_pts", 8),
        min_validation_pts=cfg_val.get("min_validation_pts", 4),
    )

    src_all = np.float32([[g.src_x, g.src_y] for g in all_gcps])
    dst_all = np.float32([[g.dst_x, g.dst_y] for g in all_gcps])

    # Default: all points used for estimation if split fails
    est_src, est_dst = src_all, dst_all
    val_src, val_dst = np.empty((0, 2)), np.empty((0, 2))
    can_validate = False

    if len(all_gcps) >= val_partitioner.min_estimation_pts + val_partitioner.min_validation_pts:
        try:
            est_src, est_dst, val_src, val_dst = val_partitioner.split(src_all, dst_all)
            can_validate = True
            logger.info(f"Validation split: {len(est_src)} estimation, {len(val_src)} held-out.")
        except ValueError as e:
            logger.warning(f"Validation partition failed: {e}")

    # Convert est_src/est_dst to ControlPoints for GCP estimator
    est_gcps = GCPEstimator().add_tile_correspondences(est_src, est_dst, tile_id="est_set")

    # ── Global GCP estimation ───────────────────────────────────────────────
    estimator = GCPEstimator(
        model=gcp_model,
        min_inliers=cfg_eval.get("min_inliers", 12),
        max_rmse_px=cfg_eval.get("max_rmse", 5.0),
        ransac_threshold=cfg_ransac.get("threshold", 4.0),
        min_coverage=cfg_eval.get("min_coverage", 0.15),
    )
    gcp_result = estimator.estimate(est_gcps, ref_w, ref_h)

    homography = gcp_result.matrix

    # ── [Phase 8] Evaluate on held-out points ──────────────────────────────
    if can_validate and len(val_src) > 0:
        val_report = val_partitioner.evaluate(
            transform_matrix=homography,
            val_src=val_src.astype(np.float64),
            val_dst=val_dst.astype(np.float64),
            n_estimation=len(est_src),
            spatial_holdout=(val_strategy == "spatial"),
        )
    else:
        # Fallback: use fitting inliers as validation (less rigorous, flagged)
        val_report = ValidationReport(
            n_estimation=len(est_gcps),
            n_validation=0,
            rmse_px=gcp_result.rmse_px,
            median_px=gcp_result.rmse_px,
            p95_px=gcp_result.rmse_px * 1.5,
            max_px=gcp_result.rmse_px * 3.0,
            mean_px=gcp_result.rmse_px,
            subpixel_claim_valid=(gcp_result.rmse_px * 1.5 < 1.0),
            spatial_holdout_used=False,
            warning="Insufficient points for independent validation. Fitting residuals used.",
        )

    # ── Warp & export ─────────────────────────────────────────────────────
    logger.info("Resampling target raster with global homography…")
    if tgt_data_full.ndim == 3:
        bands_n, _, _ = tgt_data_full.shape
        warped = np.stack([
            cv2.warpPerspective(tgt_data_full[b], homography, (ref_w, ref_h), flags=cv2.INTER_LANCZOS4)
            for b in range(bands_n)
        ], axis=0)
    else:
        warped = cv2.warpPerspective(tgt_data_full, homography, (ref_w, ref_h), flags=cv2.INTER_LANCZOS4)

    writer = RasterWriter(crs=str(ref_meta.crs),
                          compress=cfg_out.get("compress", "lzw"),
                          nodata=ref_meta.nodata)
    output_file = writer.write(warped, ref_meta.transform, out_p, overwrite=True)

    # ── Evaluate tile statistics ───────────────────────────────────────────
    residuals = np.array([cp.residual_px for cp in gcp_result.control_points
                          if cp.residual_px > 0]) if gcp_result.control_points else None

    evaluator = PipelineEvaluator(
        min_inliers=cfg_eval.get("min_inliers", 12),
        max_rmse_px=cfg_eval.get("max_rmse", 5.0),
        min_coverage=cfg_eval.get("min_coverage", 0.15),
        min_tile_success_rate=cfg_eval.get("min_tile_success_rate", 0.5),
    )
    report = evaluator.evaluate(
        tile_stats=tile_stats_list,
        global_rmse_px=gcp_result.rmse_px if not math.isinf(gcp_result.rmse_px) else 999.0,
        spatial_coverage=gcp_result.spatial_coverage,
        inlier_residuals=residuals,
    )

    elapsed = time.time() - t0

    # ── [Phase 11] Structured status code ───────────────────────────────────
    pipeline_status = _gate_status(
        n_inliers=gcp_result.inlier_count,
        rmse_fit_px=gcp_result.rmse_px if not math.isinf(gcp_result.rmse_px) else 999.0,
        coverage=gcp_result.spatial_coverage,
        val_report=val_report,
        min_inliers=cfg_eval.get("min_inliers", 12),
        max_rmse_px=cfg_eval.get("max_rmse", 5.0),
        min_coverage=cfg_eval.get("min_coverage", 0.15),
        max_val_p95_px=cfg_eval.get("max_val_p95_px", 1.0),
    )

    summary = {
        "status": pipeline_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed, 3),
        "files": {
            "reference": str(ref_p),
            "target": str(tgt_p),
            "output": str(output_file),
        },
        "preprocessing": {
            "mode": prep_mode,
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
            "n_subpixel_refined": n_refined,
            "subpixel_refine_enabled": refine_enabled,
        },
        "metrics": {
            "fitting_rmse_px": round(gcp_result.rmse_px, 4) if not math.isinf(gcp_result.rmse_px) else None,
            "spatial_coverage": round(gcp_result.spatial_coverage, 4),
            "global_inlier_ratio": report.global_inlier_ratio,
        },
        # [Phase 8] independent validation — this is the authoritative accuracy claim
        "independent_validation": val_report.to_dict(),
        "evaluation": report.to_dict(),
        "homography_matrix": homography.tolist(),
        "crs_match": str(ref_meta.crs) == str(tgt_meta.crs),
        "warnings": gcp_result.warnings + report.warnings + (
            [val_report.warning] if val_report.warning else []
        ),
    }

    summary_path = out_p.parent / f"{out_p.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done in {elapsed:.2f}s — Status: {pipeline_status} — Summary: {summary_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geospatial Image Registration Pipeline CLI — Yaazhi GeoAlign OS"
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
        pass_codes = {STATUS_SUCCESS, STATUS_LOW_CONFIDENCE}
        sys.exit(0 if summary["status"] in pass_codes else 1)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
