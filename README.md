# Yaazhi GeoAlign OS — Chandrayaan-2 Geospatial Image Registration Pipeline

[![CI](https://github.com/JS-0408/sih/actions/workflows/ci.yml/badge.svg)](https://github.com/JS-0408/sih/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

> **ISRO SIH26166** — Multi-modal, Sun-angle and scale-invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS).

---

## Overview

An end-to-end, production-grade geospatial image registration system implementing a hybrid classical + deep learning feature correspondence pipeline. Capable of aligning multi-modal lunar imagery from Chandrayaan-2's three sensors (OHRC, TMC-2, and IIRS) with sub-pixel accuracy under extreme illumination, scale, and spectral diversity.

---

## Architecture

```
Chandrayaan-2 / Sentinel-2 / Custom GeoTIFF Inputs
                    │
                    ▼
         ┌─────────────────────┐
         │   Raster I/O Layer  │  ← rasterio windowed streaming (RAM-safe)
         │   (src/io/)         │     CRS reprojection, band selection
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Spatial Tile Grid  │  ← NxN window grid (512px overlap=0.2)
         │  (main.py)          │     memory-bounded windowed reads
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Feature Engine     │  ← SIFT / ORB detection
         │  (src/processing/)  │     NxN spatial grid pruning (uniform density)
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Matching Engine    │  ← FLANN KD-tree + Lowe ratio test (0.75)
         │  (src/matching/)    │     SuperPoint + LightGlue (deep, optional)
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  RANSAC Verifier    │  ← Outlier rejection, inlier masking
         │  (src/matching/)    │     Homography / Affine / Similarity models
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  GCP Aggregator     │  ← Global GCP pooling across all tiles
         │  (src/geometry/)    │     Convex hull spatial coverage diagnostics
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Sub-Pixel Warp     │  ← cv2.warpPerspective (INTER_LINEAR)
         │  (main.py)          │     Global homography application
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Quality Gate       │  ← RMSE, inlier count, coverage checks
         │  (src/metrics/)     │     Automated pass/fail + summary.json
         └──────────┬──────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   Registered GeoTIFF    summary.json
   (data/processed/)    (metrics report)
```

---

## Sensor Compatibility Matrix

| Sensor | Type | Resolution | Status |
|:-------|:-----|:----------:|:------:|
| **OHRC** | Visible panchromatic | ~0.32 m | Supported |
| **TMC-2** | Panchromatic stereo | ~5 m | Supported |
| **IIRS** | Hyperspectral IR | ~80 m | Supported (via deep matcher) |
| **Sentinel-2** | Multispectral (real-world test) | ~10 m | Verified |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/JS-0408/sih.git
cd sih
pip install -r requirements.txt
```

### 2. Run Quick Demo (Synthetic Data, Zero Config)

```bash
py run_demo.py
# or
make demo
```

### 3. Launch Interactive Web Dashboard

```bash
py server.py
# or
make server
# Open browser: http://127.0.0.1:5000
```

### 4. Register Custom GeoTIFF Pair via CLI

```bash
py main.py \
  --reference data/raw/reference.tif \
  --target    data/raw/target.tif \
  --output    data/processed/registered.tif \
  --config    config/pipeline_config.yaml
```

### 5. Download Model Weights (Optional — Deep Matching)

```bash
py scripts/download_weights.py
# or
make weights
```

### 6. Run Full Test Suite

```bash
py -m pytest tests/ -v
# or
make test
```

---

## Benchmark Results

| Test Scenario | Resolution | Data Type | Detector | RMSE (px) | Coverage | Status |
|:-------------|:----------:|:---------:|:--------:|:---------:|:--------:|:------:|
| Synthetic Pair (1024²) | 1024×1024 | `uint8` | SIFT | **0.372** | 80.8% | PASS |
| Hard 3-Band Scene (2048²) | 2048×2048 | `uint8` RGB | SIFT | **0.419** | 75.0% | PASS |
| Real Sentinel-2 (1536²) | 1536×1536 | `uint16` | SIFT | **0.293** | 88.0% | PASS |

---

## Repository Structure

```
sih/
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── config/
│   ├── phase1_config.yaml       # Phase 1 baseline config
│   └── pipeline_config.yaml     # Master centralized config
├── data/
│   ├── raw/                     # Input rasters (git-ignored)
│   └── processed/               # Output registered files (git-ignored)
├── weights/                     # Model checkpoints (git-ignored)
├── scripts/
│   ├── download_weights.py      # Model weight downloader
│   ├── generate_synthetic_geotiff.py
│   ├── fetch_real_satellite_data.py
│   └── ray_dispatcher.py
├── src/
│   ├── geometry/gcp_estimator.py
│   ├── io/raster_loader.py + raster_writer.py
│   ├── matching/flann_matcher.py + ransac_filter.py + deep_matcher.py
│   ├── metrics/evaluator.py + rmse_calculator.py
│   └── processing/keypoint_detector.py + grid_filter.py
├── tests/                       # 71 pytest unit/integration tests
├── web/                         # Web UI dashboard (HTML/CSS/JS)
├── .gitignore
├── LICENSE                      # MIT License
├── Makefile                     # Developer shortcuts
├── README.md
├── app.py                       # Streamlit UI (optional)
├── main.py                      # CLI entrypoint
├── requirements.txt
├── run.bat                      # 1-click Windows launcher
├── run.sh                       # 1-click Linux/macOS launcher
└── server.py                    # Flask API + Web dashboard server
```

---

## Configuration

All pipeline parameters are centralized in `config/pipeline_config.yaml`:

```yaml
tiling:
  tile_size: 512
  overlap_pct: 0.20

keypoints:
  method: "SIFT"          # or "ORB"
  max_keypoints: 5000

ransac:
  model: "homography"
  threshold: 5.0

evaluation:
  min_inliers: 4
  max_rmse: 15.0
```

---

## License

MIT License — Copyright (c) 2026 Santhosh

See [LICENSE](LICENSE) for full terms.

---

## About — ISRO SIH26166

This software implements an adaptive framework for multi-modal lunar image correspondence and registration, targeting Chandrayaan-2 OHRC, TMC-2, and IIRS imagery. The system is purpose-built to handle solar illumination variance, multi-scale resolution jumps, and cross-spectral sensor differences using a hybrid classical computer vision and deep learning approach.
