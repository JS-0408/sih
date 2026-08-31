# Yaazhi GeoAlign OS — Chandrayaan-2 Geospatial Image Registration Engine

[![CI](https://github.com/JS-0408/sih/actions/workflows/ci.yml/badge.svg)](https://github.com/JS-0408/sih/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

> **ISRO SIH26166** — Robust multi-modal image correspondence and sub-pixel geospatial registration for Chandrayaan-2 optical imagery (OHRC, TMC-2, and IIRS) under solar angle variations, scale jumps, and cross-sensor offsets.

---

## Technical Highlights

- **Windowed Streaming I/O**: RAM-bounded processing (<8 GB) using `rasterio.windows.Window` for multi-gigabyte rasters.
- **Radiometric & Illumination Normalisation**: CLAHE, Laplacian gradient magnitude, and Log-CLAHE strategies to overcome extreme solar phase-angle and shadow variations.
- **Multi-Tile Spatial Coarse-to-Fine Matching**: Grid-filtered SIFT/ORB feature detection with FLANN KD-tree search and Lowe ratio verification.
- **Sub-Pixel Correspondence Refinement**: Local Normalised Cross-Correlation (NCC) peak fitting with 2D parabolic interpolation (<0.25 px residual capability).
- **Independent Validation Engine**: Spatial holdout partitioning (outermost 25% scene points) to independently verify accuracy without fitting-residual bias.
- **Evidence-Based Quality Gates**: Structured status codes (`SUCCESS`, `LOW_CONFIDENCE`, `INSUFFICIENT_CORRESPONDENCES`, `VALIDATION_FAILURE`).
- **Interactive SIH Web Console**: Vibrant, high-contrast Mission Control web interface with real-time tile diagnostic maps, split-screen overlay visualizers, and dataset presets.

---

## Canonical Pipeline Architecture

```
Ref & Tgt GeoTIFF Rasters / PDS4 Datasets
          │
          ▼
┌──────────────────────────────────────┐
│  Preflight Probe & Metadata Check    │  ← Probe CRS & dimensions
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Windowed Tile Grid Processing       │  ← RAM-safe windowed reads
│  (main.py / server.py)               │     NxN spatial grid filtering
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Illumination Normalization          │  ← CLAHE / Gradient / Log-CLAHE
│  (src/preprocessing/illumination.py) │     Shadow & solar angle invariance
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Feature Matching & RANSAC           │  ← SIFT/ORB + FLANN + RANSAC
│  (src/processing / src/matching)     │     Global GCP collection
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Sub-Pixel NCC Refinement            │  ← Parabolic peak fitting
│  (src/refinement/subpixel.py)        │     < 0.25 px local offset tuning
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Independent Spatial Validation      │  ← Outermost 25% point holdout
│  (src/metrics/validation.py)         │     P95 error gate (<1.0 px claim)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Resampling & Output Export          │  ← LANCZOS4 high-fidelity warp
│  (main.py + src/io/raster_writer.py) │     GeoTIFF + summary report
└──────────────────────────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/JS-0408/sih.git
cd sih
pip install -r requirements.txt
```

### 2. Launch Interactive Web Console

```bash
py server.py
```
Open **`http://127.0.0.1:5000`** in your browser to access the SIH Mission Control Dashboard.

### 3. Run Master CLI Pipeline

```bash
py main.py \
  --reference data/reference.tif \
  --target    data/target.tif \
  --output    outputs/registered.tif \
  --config    config/pipeline_config.yaml
```

### 4. Run Scientific Benchmark Suite

```bash
py benchmark/run.py
```

### 5. Run Full Test Suite

```bash
py -m pytest tests/ -v
```

---

## Project Structure

```
sih/
├── .github/workflows/ci.yml       # GitHub Actions automated test workflow
├── benchmark/                     # Benchmark suite & data generator
│   ├── run.py                     # Master benchmark runner
│   └── results/                   # Machine-readable CSV/JSON benchmark outputs
├── config/
│   └── pipeline_config.yaml       # Central configuration file
├── data/                          # Sample rasters & test datasets
├── outputs/                       # Registered GeoTIFF outputs & summaries
├── src/
│   ├── geometry/gcp_estimator.py  # GCP aggregation & model fitting
│   ├── io/                        # Windowed RasterLoader & RasterWriter
│   ├── matching/                  # FLANN, RANSAC, DeepMatcher backends
│   ├── metrics/
│   │   ├── evaluator.py           # Tile stats & overall gate evaluation
│   │   ├── rmse_calculator.py     # Pixel & geographic RMSE formulas
│   │   └── validation.py          # Independent holdout validation
│   ├── preprocessing/
│   │   └── illumination.py        # Radiometric normalisation
│   ├── refinement/
│   │   └── subpixel.py            # Sub-pixel NCC peak refiner
│   └── processing/                # Keypoint detection & spatial grid filter
├── web/                           # SIH Hackathon Console Frontend
│   ├── index.html                 # Main console application markup
│   ├── style.css                  # Vibrant Mission Control styling
│   └── app.js                     # Console orchestration & API client
├── tests/                         # Pytest unit & integration tests
├── server.py                      # Flask REST API backend & static file server
├── main.py                        # Canonical CLI orchestrator
└── requirements.txt
```

---

## License

MIT License — Copyright (c) 2026 Santhosh Jayakumar & Team. See [LICENSE](LICENSE) for full details.
