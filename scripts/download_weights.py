"""
scripts/download_weights.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh — MIT License

Programmatic downloader for deep learning model weights.
Downloads SuperPoint and LightGlue model checkpoints
from public repositories into the local /weights directory.

Usage:
    py scripts/download_weights.py
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WeightDownloader")

ROOT   = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"

# ---------------------------------------------------------------------------
# Registry of downloadable model weights
# ---------------------------------------------------------------------------
WEIGHT_REGISTRY: list[dict] = [
    {
        "name": "SuperPoint",
        "file": "superpoint_v1.pth",
        "url": "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
        "sha256": "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e",
        "description": "SuperPoint self-supervised keypoint detector and descriptor",
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(dest: Path, expected_sha256: str) -> None:
    actual = _sha256(dest)
    if actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"[CHECKSUM MISMATCH] {dest.name}: expected {expected_sha256}, got {actual}"
        )


def download_weight(entry: dict) -> Path | None:
    """Download one weight file if not already present."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / entry["file"]
    expected_sha256 = entry.get("sha256")
    if not expected_sha256:
        raise ValueError(f"Missing SHA-256 for {entry['name']} ({entry['file']})")

    if dest.exists():
        _verify_checksum(dest, expected_sha256)
        logger.info(f"[SKIP] Already downloaded: {dest.name}")
        return dest

    logger.info(f"[DOWNLOAD] {entry['name']} -> {dest.name}")
    logger.info(f"  Source URL: {entry['url']}")

    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(entry["url"], tmp_dest)
        _verify_checksum(tmp_dest, expected_sha256)
        shutil.move(tmp_dest, dest)
        logger.info(f"[OK] Saved to: {dest}")
    except HTTPError as exc:
        tmp_dest.unlink(missing_ok=True)
        logger.warning(
            f"[WARN] Download unavailable for {entry['name']} (HTTP {exc.code}).\n"
            f"  Please manually place '{entry['file']}' in the /weights directory."
        )
        return None
    except URLError as exc:
        tmp_dest.unlink(missing_ok=True)
        logger.warning(
            f"[WARN] Network error while downloading {entry['name']}: {exc.reason}\n"
            f"  Please manually place '{entry['file']}' in the /weights directory."
        )
        return None
    except Exception as exc:
        tmp_dest.unlink(missing_ok=True)
        logger.warning(
            f"[WARN] Could not download {entry['name']}: {exc}\n"
            f"  Please manually place '{entry['file']}' in the /weights directory."
        )
        return None

    return dest


def main() -> None:
    logger.info("=" * 55)
    logger.info("  Yaazhi GeoAlign OS — Model Weight Downloader")
    logger.info("=" * 55)
    logger.info(f"  Weights directory: {WEIGHTS_DIR}")

    for entry in WEIGHT_REGISTRY:
        download_weight(entry)

    # Create weights/README.md if missing
    readme = WEIGHTS_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Model Weights\n\n"
            "This directory stores downloaded PyTorch model checkpoint files.\n\n"
            "**Files in this directory are excluded from Git tracking** (see `.gitignore`).\n\n"
            "To download weights automatically, run:\n\n"
            "```bash\n"
            "py scripts/download_weights.py\n"
            "```\n\n"
            "## Included Models\n\n"
            "| Model | File | Source |\n"
            "|:------|:-----|:-------|\n"
            "| SuperPoint | `superpoint_v1.pth` | Magic Leap GitHub |\n",
            encoding="utf-8",
        )

    logger.info("[DONE] Weight download process complete.")


if __name__ == "__main__":
    main()
