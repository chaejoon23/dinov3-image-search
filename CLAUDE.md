# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DINOv3 testing environment containing Meta AI's DINOv3 vision foundation model implementation. The repository includes:
- Local clone of the official DINOv3 repository (`dinov3/`)
- Simple smoke test script (`smoke_test.py`) for model validation
- Python virtual environment for isolated dependency management

## Architecture

### Repository Structure

```
dinov3_test/
├── dinov3/              # Official DINOv3 repository (cloned)
│   ├── dinov3/          # Core package modules
│   │   ├── hub/         # PyTorch Hub loading interface
│   │   ├── models/      # Vision Transformer & ConvNeXt implementations
│   │   ├── layers/      # Attention, FFN, normalization layers
│   │   ├── train/       # Training scripts and configs
│   │   ├── eval/        # Evaluation tasks (classification, segmentation, depth)
│   │   └── data/        # Dataset handling
│   ├── hubconf.py       # PyTorch Hub entry point
│   └── requirements.txt # DINOv3 dependencies
├── smoke_test.py        # Local validation script
└── venv/                # Python virtual environment
```

### Model Loading Architecture

The project uses **PyTorch Hub's local loading mechanism**:

1. **Hub Configuration** (`dinov3/hubconf.py`): Defines available models
2. **Backbone Modules** (`dinov3/dinov3/hub/backbones.py`): Model constructors
3. **Local Loading**: `torch.hub.load("./dinov3", "model_name", source="local", pretrained=False)`

**Key Architectural Decision**: The smoke test loads models **without pretrained weights** (`pretrained=False`) because:
- DINOv3 weights are gated and require Meta's download approval
- Structural validation doesn't require pretrained parameters
- Enables rapid testing without multi-GB downloads

### Available Models

**Vision Transformers (ViT)**: `dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`, `dinov3_vitl16`, `dinov3_vith16plus`, `dinov3_vit7b16`

**ConvNeXt**: `dinov3_convnext_tiny`, `dinov3_convnext_small`, `dinov3_convnext_base`, `dinov3_convnext_large`

**Task-Specific Heads**: `dinov3_vit7b16_lc` (classifier), `dinov3_vit7b16_dd` (depth), `dinov3_vit7b16_de` (detection), `dinov3_vit7b16_ms` (segmentation)

## Development Commands

### Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install DINOv3 package in editable mode
pip install -e ./dinov3

# Install additional dependencies
pip install torchmetrics termcolor transformers
```

### Testing

```bash
# Run smoke test
python smoke_test.py
```

**Expected Output**: Model loads, moves to device (MPS/CUDA/CPU), performs forward pass, outputs feature shape `[1, 384]` for ViT-S/16.

### Model Loading Patterns

**Without Pretrained Weights** (for structural testing):
```python
model = torch.hub.load(
    "./dinov3",
    "dinov3_vits16",
    source="local",
    pretrained=False
)
```

**With Pretrained Weights** (requires Meta approval):
```python
model = torch.hub.load(
    "./dinov3",
    "dinov3_vits16",
    source="local",
    weights="<CHECKPOINT/URL/OR/PATH>"
)
```

**Via Hugging Face** (alternative, also gated):
```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
```

### Device Compatibility

The `smoke_test.py` implements device detection priority:
1. **MPS** (Apple Silicon) - Primary target for this environment
2. **CUDA** (NVIDIA GPUs)
3. **CPU** (fallback)

## Important Conventions

### Model Weight Access

- Pretrained weights require acceptance through [Meta's download portal](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
- Use `wget` for downloading weights (browser downloads may fail)
- Weights can be passed as local paths or URLs via the `weights` parameter

### Image Preprocessing

**For LVD-1689M weights** (web images):
```python
normalize = v2.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

**For SAT-493M weights** (satellite imagery):
```python
normalize = v2.Normalize(
    mean=(0.430, 0.411, 0.296),
    std=(0.213, 0.156, 0.143)
)
```

### Module Imports

Always prefix commands with `PYTHONPATH=.` when running DINOv3 scripts:
```bash
PYTHONPATH=. python dinov3/eval/knn.py ...
```

## Known Issues & Solutions

### Issue: Missing Dependencies
**Symptoms**: `ModuleNotFoundError` for `torchmetrics`, `termcolor`, `transformers`
**Solution**: `pip install torchmetrics termcolor transformers`

### Issue: HTTP 403 Forbidden
**Symptoms**: Weight download fails with 403 error
**Solution**: Model weights are gated. Either:
1. Request access through Meta's portal and use approved URLs
2. Use `pretrained=False` for structural testing
3. Download weights via `wget` with approved URLs

### Issue: Hugging Face Gated Repo
**Symptoms**: "You are trying to access a gated repo"
**Solution**:
1. Create Hugging Face account
2. Request access to DINOv3 models
3. Login via `huggingface-cli login`

## Reference Documentation

- **Paper**: [DINOv3 (arXiv:2508.10104)](https://arxiv.org/abs/2508.10104)
- **Official Website**: https://ai.meta.com/dinov3/
- **Hugging Face Collection**: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009
- **DINOv3 README**: `dinov3/README.md` (comprehensive guide to training and evaluation)
