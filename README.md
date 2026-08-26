# Phase 1 — Geospatial Image Processing Pipeline

Distributed, modular pipeline for GeoTIFF keypoint detection, FLANN matching, RANSAC geometric filtering, and RMSE evaluation — powered by Ray for multi-node distributed compute.

---

## Architecture

```
sih/
├── config/
│   └── phase1_config.yaml      # Central parameter control
├── src/
│   ├── io/
│   │   └── raster_loader.py    # GeoTIFF loader, CRS validation, tile generator
│   ├── processing/
│   │   ├── keypoint_detector.py  # SIFT / ORB detection
│   │   └── grid_filter.py        # Spatial grid pruning
│   ├── matching/
│   │   ├── flann_matcher.py      # FLANN + Lowe's ratio test
│   │   └── ransac_filter.py      # Homography RANSAC inlier filter
│   └── metrics/
│       └── rmse_calculator.py    # Pixel + geographic RMSE
├── scripts/
│   └── test_ray_cluster.py     # Ray cluster diagnostic
├── tests/                      # pytest unit tests
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Activate virtualenv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all tests
pytest tests/ -v

# 4. Start Ray (Laptop 1 — head node)
ray start --head --port=6379

# 4b. Join worker (Laptop 2)
ray start --address='<LAPTOP1_IP>:6379'

# 5. Verify cluster
python scripts/test_ray_cluster.py
```

---

## Configuration

All pipeline parameters live in `config/phase1_config.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `tiling` | `tile_size` | 512 | Tile dimensions (px) |
| `tiling` | `overlap_pct` | 0.2 | Tile overlap fraction |
| `keypoints` | `method` | SIFT | Detector backend |
| `keypoints` | `max_keypoints` | 5000 | Max per tile |
| `keypoints` | `grid_cells` | 16 | Spatial pruning grid |
| `flann` | `trees` | 5 | KD-tree count |
| `flann` | `ratio_threshold` | 0.75 | Lowe's ratio |
| `ransac` | `threshold` | 5.0 | Reprojection error (px) |
| `geospatial` | `crs_target` | EPSG:4326 | Required CRS |
| `ray` | `head_ip` | auto | Ray head address |

---

## Ray Cluster Setup

| Node | Role | Command |
|---|---|---|
| Laptop 1 | Head | `ray start --head --port=6379` |
| Laptop 2 | Worker | `ray start --address='<LAPTOP1_IP>:6379'` |

Set `RAY_HEAD=<LAPTOP1_IP>:6379` to override the auto-detect in the diagnostic script.

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v --tb=short
```

---

## Tech Stack

- **Python** 3.11+ with full type hints
- **OpenCV** — SIFT/ORB keypoint detection, FLANN matching, RANSAC
- **Rasterio** — GeoTIFF I/O and CRS handling
- **NumPy / Shapely** — Array ops and geometric primitives
- **PyTorch** — Deep learning (Phase 2+)
- **Ray** — Distributed orchestration across dual-laptop cluster
- **pytest** — Unit test suite

---

## Phase Roadmap

| Phase | Goal |
|---|---|
| **1** ✅ | Pipeline base: I/O, keypoints, matching, metrics, Ray cluster |
| 2 | Tile-level Ray actors for distributed keypoint extraction |
| 3 | Deep feature matching (SuperPoint / LoFTR) |
| 4 | GCP registration and orthorectification |
| 5 | Full accuracy evaluation & reporting |
