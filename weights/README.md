# Model Weights

This directory stores downloaded PyTorch model checkpoint files.

**Files in this directory are excluded from Git tracking** (see `.gitignore`).

To download weights automatically run:

```bash
py scripts/download_weights.py
# or
make weights
```

## Supported Models

| Model | File | Source |
|:------|:-----|:-------|
| SuperPoint | `superpoint_v1.pth` | [Magic Leap GitHub](https://github.com/magicleap/SuperPointPretrainedNetwork) |
| LightGlue | Via `torch.hub` | [ETH CVG](https://github.com/cvg/LightGlue) |
