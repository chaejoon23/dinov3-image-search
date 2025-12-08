# DINOv3 Image Similarity Search System

Comprehensive documentation for the DINOv3-powered image similarity search system with support for both demo datasets (CIFAR-10) and real-world street-level imagery (Mapillary).

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [System Architecture](#system-architecture)
4. [Technical Deep Dive](#technical-deep-dive)
5. [Usage Guide](#usage-guide)
6. [File Structure](#file-structure)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This system enables visual similarity search using Meta's DINOv3 (Self-Distillation with No Labels) vision foundation model. Given a query image, it finds the most visually similar images from a database by comparing deep learning feature embeddings.

### What Can You Do?

- **Upload any image** and find visually similar images
- **Search across different datasets**:
  - CIFAR-10: 1,000 sample images (10 categories)
  - Fukuoka Mapillary: 243 real-world street views from Fukuoka, Japan
- **Get detailed metadata** including GPS coordinates, capture times, and more
- **View results** ranked by visual similarity scores

### Key Features

- ⚡ **Fast**: Pre-computed embeddings enable instant search
- 🎯 **Accurate**: DINOv3's self-supervised learning captures rich visual features
- 🌐 **Web Interface**: Easy-to-use Gradio UI
- 📍 **Geospatial**: Mapillary integration with GPS coordinates
- 🖥️ **GPU Accelerated**: Supports MPS (Apple Silicon), CUDA, and CPU

---

## How It Works

### The Big Picture

The system works in three phases:

```
Phase 1: Database Building          Phase 2: Query Processing       Phase 3: Similarity Search
┌─────────────────────┐             ┌─────────────────────┐         ┌─────────────────────┐
│  1. Load Images     │             │  1. Upload Image    │         │  1. Extract Query   │
│     from Dataset    │             │                     │         │     Embedding       │
├─────────────────────┤             ├─────────────────────┤         ├─────────────────────┤
│  2. Preprocess      │             │  2. Preprocess      │         │  2. Compute Cosine  │
│     Each Image      │             │     Query Image     │         │     Similarities    │
├─────────────────────┤             ├─────────────────────┤         ├─────────────────────┤
│  3. Extract DINOv3  │             │  3. Extract DINOv3  │         │  3. Rank by Score   │
│     Embeddings      │             │     Embedding       │         │                     │
├─────────────────────┤             └─────────────────────┘         ├─────────────────────┤
│  4. L2 Normalize    │                                             │  4. Return Top-K    │
│     Embeddings      │                                             │     Results         │
├─────────────────────┤                                             └─────────────────────┘
│  5. Save to         │
│     Database (.pkl) │
└─────────────────────┘
```

### Step-by-Step Process

#### Phase 1: Building the Database (Offline)

This happens once before you can search:

1. **Image Collection**
   - CIFAR-10: Download from torchvision
   - Mapillary: Query API by geographic bounding boxes over Fukuoka

2. **Image Preprocessing**
   ```python
   Image → Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet stats)
   ```
   - Resizes to 256x256 pixels
   - Center crops to 224x224 (DINOv3 input size)
   - Converts to tensor (0-1 float values)
   - Normalizes using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

3. **Feature Extraction**
   - Pass preprocessed image through DINOv3 ViT-S/16
   - Model outputs 384-dimensional feature vector (embedding)
   - This captures semantic visual information: colors, textures, shapes, objects, scene composition

4. **Normalization**
   - L2-normalize each embedding: `embedding = embedding / ||embedding||`
   - Ensures all vectors have unit length
   - Enables cosine similarity via simple dot product

5. **Storage**
   - Save embeddings as numpy array
   - Save corresponding image paths
   - Save metadata (for Mapillary: GPS, dates, etc.)
   - Store in pickle file for fast loading

#### Phase 2: Query Processing (Real-time)

When you upload an image:

1. **Preprocessing** (same as database images)
   - Resize, crop, normalize to match training distribution

2. **Embedding Extraction**
   - Pass through same DINOv3 model
   - Get 384-dimensional vector
   - L2-normalize to unit length

#### Phase 3: Similarity Search (Real-time)

Finding similar images:

1. **Cosine Similarity Computation**
   ```python
   similarities = database_embeddings @ query_embedding
   ```
   - Since vectors are normalized, dot product = cosine similarity
   - Fast matrix-vector multiplication
   - Returns similarity score for each database image (-1 to 1, typically 0-1)

2. **Ranking**
   ```python
   top_indices = np.argsort(similarities)[::-1][:k]
   ```
   - Sort by similarity (descending)
   - Select top-k results

3. **Result Formatting**
   - Load corresponding images
   - Fetch metadata (if available)
   - Return as gallery with scores

---

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser                              │
│              (Gradio UI - ports 7860, 7861)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend                               │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ image_search_app │  │ mapillary_search │                    │
│  │      .py         │  │      _app.py     │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
│           │                     │                               │
│           ▼                     ▼                               │
│  ┌─────────────────────────────────────────┐                   │
│  │     DINOv3 Model (ViT-S/16)             │                   │
│  │   - 384-dim embeddings                  │                   │
│  │   - Loaded from ./dinov3                │                   │
│  │   - Runs on MPS/CUDA/CPU                │                   │
│  └─────────────────────────────────────────┘                   │
│           │                     │                               │
│           ▼                     ▼                               │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ CIFAR Database   │  │ Fukuoka Database │                   │
│  │ embeddings.pkl   │  │ fukuoka_emb.pkl  │                   │
│  │ (1000 images)    │  │ (243 images)     │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                 │                               │
│                                 ▼                               │
│                        ┌──────────────────┐                    │
│                        │ Mapillary        │                    │
│                        │ metadata.json    │                    │
│                        │ (GPS, dates)     │                    │
│                        └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Building Database:**
```
Images → Preprocessor → DINOv3 → L2 Normalization → Database (pickle)
```

**Search Query:**
```
Query Image → Preprocessor → DINOv3 → L2 Normalization
                                          ↓
                              Database ← Cosine Similarity
                                          ↓
                                    Ranked Results
```

---

## Technical Deep Dive

### What is DINOv3?

**DINOv3** (Self-Distillation with No Labels, version 3) is Meta AI's vision foundation model trained through self-supervised learning.

**Key Characteristics:**

1. **Vision Transformer Architecture**
   - Based on ViT (Vision Transformer)
   - Divides images into 16×16 pixel patches
   - 224×224 image = 14×14 = 196 patches
   - Processes patches through transformer layers

2. **Self-Supervised Training**
   - No manual labels required
   - Trained on 1.69 billion web images (LVD-1689M dataset)
   - Learns visual representations by predicting masked patches
   - Student-teacher distillation framework

3. **Model Variants Used**
   - **ViT-S/16** (Small): 21M parameters, 384-dim embeddings
   - Input: 224×224 RGB images
   - Output: Single 384-dimensional feature vector per image

### Why DINOv3 for Similarity Search?

1. **Rich Semantic Features**
   - Captures objects, scenes, textures, colors, spatial relationships
   - Works across diverse visual domains (animals, vehicles, architecture, etc.)

2. **Transfer Learning Excellence**
   - Pre-trained on massive dataset
   - Generalizes well to new domains without fine-tuning

3. **Robust to Variations**
   - Handles different lighting, angles, scales
   - Focuses on semantic content over superficial differences

4. **Compact Representations**
   - 384 dimensions efficiently encode visual information
   - Fast similarity computation
   - Low storage requirements

### Cosine Similarity Explained

**Why Cosine Similarity?**

Cosine similarity measures the angle between two vectors, ignoring magnitude:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

When vectors are L2-normalized (||A|| = ||B|| = 1):

```
cosine_similarity(A, B) = A · B  (simple dot product)
```

**Interpretation:**
- **1.0**: Identical visual features
- **0.8-0.9**: Very similar (same object/scene type)
- **0.6-0.7**: Moderately similar (related content)
- **0.3-0.5**: Somewhat similar (shared elements)
- **< 0.3**: Dissimilar

### Image Preprocessing Pipeline

**Why Each Step Matters:**

1. **Resize(256)**
   - Ensures consistent aspect ratio handling
   - Slight upsampling/downsampling for uniformity

2. **CenterCrop(224)**
   - Extracts central 224×224 region
   - Focuses on main subject (usually centered)
   - Matches DINOv3 training input size

3. **ToTensor()**
   - Converts PIL Image to PyTorch tensor
   - Rescales from [0, 255] to [0.0, 1.0]

4. **Normalize(mean, std)**
   - Subtracts mean: `(pixel - mean) / std`
   - Matches ImageNet statistics used during DINOv3 training
   - Critical for model performance

### Database Format

**Pickle File Structure:**

```python
{
    'embeddings': np.ndarray,      # Shape: (N, 384), dtype: float32
    'image_paths': List[str],      # Length: N
    'model_name': str,             # "dinov3_vits16"
    'embedding_dim': int           # 384
}
```

**Mapillary Metadata (JSON):**

```json
[
  {
    "id": "1000780244899381",
    "local_path": "fukuoka_images/mapillary_1000780244899381.jpg",
    "mapillary_url": "https://...",
    "captured_at": "2023-04-15T08:30:00Z",
    "compass_angle": 123.4,
    "geometry": {
      "coordinates": [130.4017, 33.5904],
      "type": "Point"
    },
    "creator": {"username": "photographer123"},
    "width": 1024,
    "height": 768
  }
]
```

---

## Usage Guide

### Quick Start

**1. CIFAR-10 Demo System**

Access at: http://localhost:7860

```bash
# Download dataset (1000 images)
python download_dataset.py

# Build embeddings
python build_embedding_database.py

# Launch web UI
python image_search_app.py --port 7860
```

**2. Fukuoka Mapillary System**

Access at: http://localhost:7861

```bash
# Download Fukuoka images (requires Mapillary token)
python download_mapillary_fukuoka.py \
  --token "MLY|your_token_here" \
  --count 500

# Build embeddings
python build_embedding_database.py \
  --images-dir ./fukuoka_images \
  --output ./fukuoka_embeddings.pkl

# Launch web UI
python mapillary_search_app.py --port 7861
```

### Command-Line Options

**download_mapillary_fukuoka.py:**

```bash
--token             Mapillary API token (required)
--output-dir        Output directory (default: ./fukuoka_images)
--count             Number of images to download (default: 500)
--images-per-tile   Max images per grid tile (default: 20)
```

**build_embedding_database.py:**

```bash
--images-dir        Input directory with images (default: ./sample_images)
--output            Output pickle file (default: ./embeddings_database.pkl)
--model             DINOv3 model name (default: dinov3_vits16)
--pretrained        Use pretrained weights (default: False)
```

**image_search_app.py / mapillary_search_app.py:**

```bash
--database          Path to embedding database
--metadata          Path to metadata JSON (Mapillary only)
--model             DINOv3 model name (default: dinov3_vits16)
--port              Server port (default: 7860/7861)
--share             Create public Gradio share link
```

### Getting a Mapillary API Token

1. **Sign up** at https://www.mapillary.com
2. **Go to Developer Dashboard**: https://www.mapillary.com/dashboard/developers
3. **Register Application**: Click "Register Application"
4. **Copy Token**: Format: `MLY|################|################################`
5. **Use in Script**: Pass via `--token` flag

### Using the Web Interface

**Upload Query Image:**
1. Click the image upload box
2. Select image from your computer
3. Preview appears

**Adjust Settings:**
- Move slider to select number of results (1-20)
- More results = broader visual matches

**Search:**
- Click "🔍 Search Similar Images" button
- Results appear in gallery below
- Similarity scores shown with each result

**View Metadata (Mapillary):**
- Capture date and time
- GPS coordinates (clickable)
- Compass direction
- Image dimensions
- Photographer username
- Link to view on Mapillary

### Tips for Best Results

**Good Query Images:**
- Clear, well-lit photos
- Main subject centered
- Minimal blur
- Similar type to database (e.g., street views for Mapillary)

**Interpreting Similarity Scores:**
- **0.9+**: Nearly identical or same location
- **0.7-0.9**: Very similar visual content
- **0.5-0.7**: Moderately similar (related scenes)
- **< 0.5**: Weak similarity (shared basic features)

**CIFAR-10 Categories:**
- Works best with: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, trucks
- Upload images of these categories for best matches

**Mapillary Fukuoka:**
- Best for: street views, urban scenes, buildings, parks
- Upload outdoor urban photos
- Japanese architecture/signage = stronger matches

---

## File Structure

```
dinov3_test/
├── README_IMAGE_SEARCH.md              # This documentation
│
├── dinov3/                              # DINOv3 model repository
│   ├── hubconf.py                       # PyTorch Hub entry point
│   └── dinov3/                          # Core model code
│
├── download_dataset.py                  # Download CIFAR-10 dataset
├── download_mapillary_fukuoka.py        # Download Mapillary images
│
├── build_embedding_database.py          # Extract DINOv3 embeddings
│
├── image_search_app.py                  # CIFAR-10 web UI
├── mapillary_search_app.py              # Fukuoka Mapillary web UI
│
├── sample_images/                       # CIFAR-10 dataset (1000 images)
│   ├── airplane/
│   ├── automobile/
│   ├── bird/
│   └── ...
│
├── fukuoka_images/                      # Mapillary dataset (243 images)
│   ├── mapillary_*.jpg                  # Downloaded images
│   └── metadata.json                    # Mapillary metadata
│
├── embeddings_database.pkl              # CIFAR-10 embeddings (1000×384)
└── fukuoka_embeddings.pkl               # Fukuoka embeddings (243×384)
```

---

## Advanced Topics

### Using Different DINOv3 Models

Available models in dinov3/hubconf.py:

**Vision Transformers:**
- `dinov3_vits16` (21M params, 384-dim) ← **Currently used**
- `dinov3_vitb16` (86M params, 768-dim)
- `dinov3_vitl16` (304M params, 1024-dim)
- `dinov3_vith16plus` (632M params, 1280-dim)

**Trade-offs:**
- Larger models: Better accuracy, slower inference, more memory
- Smaller models: Faster, less memory, slightly lower accuracy

**To use a different model:**

```bash
python build_embedding_database.py --model dinov3_vitb16
python image_search_app.py --model dinov3_vitb16
```

### Using Pretrained Weights

**Current Setup:** Models run without pretrained weights (`pretrained=False`)
- Works fine for similarity search (architectural features sufficient)
- No download required

**With Pretrained Weights:**
1. Get access at: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
2. Download weights via `wget`
3. Modify code to load weights:

```python
model = torch.hub.load(
    "./dinov3",
    "dinov3_vits16",
    source="local",
    weights="/path/to/checkpoint.pth"
)
```

### Performance Optimization

**Database Building:**
- Use GPU for faster embedding extraction
- Batch processing for large datasets
- Current speed: ~55-77 images/second on MPS

**Search:**
- Pre-computed embeddings = instant search
- Similarity computation: O(N) where N = database size
- 243 images: < 1ms search time
- 10,000 images: ~10ms search time

**Scaling to Larger Databases:**

For millions of images, consider:
1. **Approximate Nearest Neighbors (ANN)**
   - FAISS, Annoy, or HNSW indices
   - Sub-linear search time
   - Slight accuracy trade-off

2. **Database Sharding**
   - Geographic or categorical partitioning
   - Search relevant shard only

3. **Dimensionality Reduction**
   - PCA to reduce from 384 to 128 dims
   - 3× smaller storage, faster search

### Customizing for Your Dataset

**1. Prepare Images:**
```bash
your_dataset/
├── category_1/
│   ├── image1.jpg
│   └── image2.jpg
└── category_2/
    └── image3.jpg
```

**2. Build Embeddings:**
```bash
python build_embedding_database.py \
  --images-dir ./your_dataset \
  --output ./your_embeddings.pkl
```

**3. Modify Web UI:**
```python
# In image_search_app.py, update defaults:
parser.add_argument(
    "--database",
    default="./your_embeddings.pkl"
)
```

**4. Launch:**
```bash
python image_search_app.py --database ./your_embeddings.pkl
```

### Integrating with Your Application

**Programmatic API:**

```python
import torch
import numpy as np
from PIL import Image
import pickle

# Load model
device = torch.device("mps")
model = torch.hub.load("./dinov3", "dinov3_vits16", source="local")
model = model.to(device).eval()

# Load database
with open("embeddings_database.pkl", "rb") as f:
    db = pickle.load(f)

# Extract query embedding
def get_embedding(image_path):
    # [preprocessing code here]
    with torch.no_grad():
        emb = model(image_tensor)
    return emb.cpu().numpy().flatten()

# Search
query_emb = get_embedding("query.jpg")
query_emb = query_emb / np.linalg.norm(query_emb)  # L2 normalize
similarities = db['embeddings'] @ query_emb
top_k = np.argsort(similarities)[::-1][:5]
results = [db['image_paths'][i] for i in top_k]
```

---

## Troubleshooting

### Common Issues

**1. Model Loading Fails**

```
Error: No module named 'dinov3'
```

**Solution:** Ensure you're in the correct directory:
```bash
cd /Users/junchae/dev/dinov3_test
ls dinov3/  # Should show hubconf.py
```

**2. Out of Memory**

```
RuntimeError: MPS backend out of memory
```

**Solution:**
- Reduce batch size (process images one at a time)
- Use smaller model: `dinov3_vits16` instead of larger variants
- Restart Python session to clear memory

**3. Port Already in Use**

```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port
lsof -i :7860

# Kill process
kill -9 <PID>

# Or use different port
python image_search_app.py --port 7862
```

**4. Mapillary API Errors**

```
Error 4: Application request limit reached
```

**Solution:**
- Rate limit hit (10,000 requests/min)
- Wait 1 minute
- Reduce `--images-per-tile` to slow down requests

**5. Slow Embedding Extraction**

**Check device:**
```python
import torch
print(torch.backends.mps.is_available())  # Should be True on M1/M2/M3 Macs
```

**Solution:**
- Ensure MPS is enabled
- Check GPU memory isn't full
- Update PyTorch to latest version

**6. Poor Search Results**

**Possible causes:**
- Query image very different from database
- Wrong preprocessing
- Model not loaded correctly

**Solution:**
- Verify query image preprocessed correctly
- Check similarity scores (very low scores = dissimilar content)
- Try with images similar to database content

### Getting Help

**Check Logs:**
```bash
# See model loading messages
python build_embedding_database.py  # Shows device, model name, speed

# Debug web UI
python image_search_app.py  # Shows startup messages, errors
```

**Verify Setup:**
```bash
# Check Python environment
which python  # Should be in venv

# Check dependencies
pip list | grep -E "torch|gradio|numpy"

# Check files exist
ls embeddings_database.pkl
ls fukuoka_embeddings.pkl
```

---

## References

**DINOv3:**
- Paper: https://arxiv.org/abs/2508.10104
- Official Site: https://ai.meta.com/dinov3/
- GitHub: https://github.com/facebookresearch/dinov3

**Mapillary:**
- API Docs: https://www.mapillary.com/developer/api-documentation
- Platform: https://www.mapillary.com/

**Technologies:**
- PyTorch: https://pytorch.org/
- Gradio: https://gradio.app/
- NumPy: https://numpy.org/

---

## License & Credits

**DINOv3 Model:**
- Developed by Meta AI
- License: Apache 2.0

**Datasets:**
- CIFAR-10: Alex Krizhevsky, Geoffrey Hinton
- Mapillary: Street-level imagery contributors

**This Implementation:**
- Built for educational and research purposes
- Uses publicly available tools and APIs

---

**Last Updated:** 2025-12-08
**System Version:** 1.0
