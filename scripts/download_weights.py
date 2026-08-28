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
import sys
import urllib.request
from pathlib import Path

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
        "sha256": None,  # skip hash verification for demo
        "description": "SuperPoint self-supervised keypoint detector and descriptor",
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weight(entry: dict) -> Path:
    """Download one weight file if not already present."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / entry["file"]

    if dest.exists():
        logger.info(f"[SKIP] Already downloaded: {dest.name}")
        return dest

    logger.info(f"[DOWNLOAD] {entry['name']} -> {dest.name}")
    logger.info(f"  Source URL: {entry['url']}")

    try:
        urllib.request.urlretrieve(entry["url"], dest)
        logger.info(f"[OK] Saved to: {dest}")
    except Exception as exc:
        logger.warning(
            f"[WARN] Could not download {entry['name']}: {exc}\n"
            f"  Please manually place '{entry['file']}' in the /weights directory."
        )
        return dest

    if entry.get("sha256"):
        actual = _sha256(dest)
        if actual != entry["sha256"]:
            logger.error(f"[CHECKSUM MISMATCH] Expected {entry['sha256']}, got {actual}")
            dest.unlink()
            raise RuntimeError(f"Weight file {entry['file']} is corrupted.")
        logger.info(f"[OK] Checksum verified.")

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
