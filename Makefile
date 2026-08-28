# =============================================================================
# Makefile — Yaazhi GeoAlign OS / SIH26166 Developer Shortcuts
# Copyright (c) 2026 Santhosh — MIT License
# =============================================================================

PYTHON      = py
CONFIG      = config/pipeline_config.yaml
REF         = data/raw/reference.tif
TGT         = data/raw/target.tif
OUT         = data/processed/registered_output.tif

.PHONY: help install test demo server weights clean lint

help:  ## Show available make targets
	@echo ""
	@echo "  Yaazhi GeoAlign OS — Developer Makefile"
	@echo "  ========================================="
	@echo "  make install   - Install all Python dependencies"
	@echo "  make test      - Run full pytest suite (71 tests)"
	@echo "  make demo      - Run quick CLI demo on synthetic data"
	@echo "  make server    - Launch interactive Web UI dashboard"
	@echo "  make weights   - Download model weights (SuperPoint/LightGlue)"
	@echo "  make clean     - Remove cached files and build artifacts"
	@echo "  make lint      - Run code style checks"
	@echo ""

install:  ## Install all dependencies from requirements.txt
	$(PYTHON) -m pip install -r requirements.txt

test:  ## Run full automated test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

demo:  ## Run zero-configuration quick CLI demo
	$(PYTHON) run_demo.py

server:  ## Launch interactive Web UI dashboard at http://127.0.0.1:5000
	$(PYTHON) server.py

weights:  ## Download model weights programmatically
	$(PYTHON) scripts/download_weights.py

register:  ## Register a custom raster pair (set REF= TGT= OUT= variables)
	$(PYTHON) main.py --reference $(REF) --target $(TGT) --output $(OUT) --config $(CONFIG)

clean:  ## Remove pycache, pytest artifacts
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@if exist .pytest_cache rd /s /q .pytest_cache
	@echo "[clean] Done."

lint:  ## Run basic code style check via pyflakes
	$(PYTHON) -m pyflakes src/ main.py server.py
