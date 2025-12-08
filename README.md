# DINOv3 Image Similarity Search System

Visual similarity search system powered by Meta's DINOv3 vision foundation model. Find similar images using deep learning embeddings and cosine similarity.

---

## 🚀 Quick Start

### Launch Web Interfaces

**CIFAR-10 Demo** (1,000 sample images):
```bash
python image_search_app.py --port 7860
```
Access at: http://localhost:7860

**Fukuoka Mapillary** (243 street-level images):
```bash
python mapillary_search_app.py --port 7861
```
Access at: http://localhost:7861

---

## ✨ Features

### Two Complete Search Systems

**1. CIFAR-10 Demo System**
- 1,000 images across 10 categories
- Quick demonstration and testing
- Categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, trucks

**2. Fukuoka Mapillary System**
- 243 real-world street-level images from Fukuoka, Japan
- GPS coordinates, capture dates, compass directions
- Direct links to view on Mapillary
- Rich metadata for each result

### Core Capabilities

- ⚡ **Instant Search**: Pre-computed embeddings enable sub-second search
- 🎯 **Accurate Matching**: DINOv3's self-supervised learning captures rich visual features
- 🌐 **Easy to Use**: Web-based Gradio interface
- 📍 **Geospatial**: GPS coordinates and location metadata
- 🖥️ **GPU Accelerated**: Supports MPS (Apple Silicon), CUDA, and CPU

---

## 🎯 What Can You Do?

1. **Upload any image** to find visually similar images
2. **Search across datasets**: CIFAR-10 demo or real-world Mapillary imagery
3. **View similarity scores** ranging from 0 to 1
4. **Access metadata**: GPS location, capture time, image details
5. **Download more data**: Expand Mapillary coverage to other cities

---

## 📊 System Specifications

**Model:** DINOv3 ViT-S/16
- 21 million parameters
- 384-dimensional embeddings
- Self-supervised training on 1.69B images

**Performance:**
- Embedding extraction: ~55-77 images/second (Apple Silicon MPS)
- Search time: < 1ms for 1,000 images
- GPU/CPU support: MPS, CUDA, or CPU fallback

**Databases:**
- CIFAR-10: 1,000 images, 1.46 MB embeddings
- Fukuoka: 243 images, 0.36 MB embeddings

---

## 📚 Documentation

### Start Here
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation index and navigation guide

### Quick References
- **[QUICK_START.md](QUICK_START.md)** - Commands, troubleshooting, quick reference
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Visual diagrams and system architecture

### Complete Guide
- **[README_IMAGE_SEARCH.md](README_IMAGE_SEARCH.md)** - Comprehensive technical documentation
  - How it works (step-by-step)
  - Technical deep dive into DINOv3
  - Usage guide with all options
  - Advanced topics and customization
  - Troubleshooting

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Optional: GPU (MPS/CUDA) for faster processing

### Environment Setup
```bash
# Already set up in this project
source venv/bin/activate

# Verify installation
pip list | grep -E "torch|gradio|numpy"
```

### Download Sample Data
```bash
# CIFAR-10 dataset (1000 images)
python download_dataset.py

# Fukuoka Mapillary images (requires API token)
python download_mapillary_fukuoka.py --token "YOUR_TOKEN" --count 500
```

### Build Embeddings
```bash
# CIFAR-10
python build_embedding_database.py --images-dir ./sample_images

# Fukuoka Mapillary
python build_embedding_database.py \
  --images-dir ./fukuoka_images \
  --output ./fukuoka_embeddings.pkl
```

---

## 📖 How It Works

### Three-Phase Process

**Phase 1: Database Building** (Offline)
1. Download images from datasets
2. Preprocess: Resize → CenterCrop → Normalize
3. Extract 384-dim embeddings via DINOv3
4. L2-normalize for cosine similarity
5. Save to pickle database

**Phase 2: Query Processing** (Real-time)
1. User uploads query image
2. Apply same preprocessing
3. Extract DINOv3 embedding
4. L2-normalize query vector

**Phase 3: Similarity Search** (Real-time)
1. Compute cosine similarity: `database @ query`
2. Sort by similarity score (descending)
3. Return top-K results with metadata

### DINOv3 Model
- Vision Transformer (ViT) architecture
- Divides image into 16×16 pixel patches
- Self-supervised learning (no manual labels)
- Captures semantic visual features: objects, scenes, textures, colors

### Cosine Similarity
```python
# When vectors are L2-normalized:
similarity = embedding1 @ embedding2

# Score interpretation:
# 0.9+   : Nearly identical
# 0.7-0.9: Very similar
# 0.5-0.7: Moderately similar
# < 0.5  : Dissimilar
```

---

## 🎨 Usage Examples

### Web Interface
1. Open http://localhost:7860 or http://localhost:7861
2. Click "Upload Query Image"
3. Adjust "Number of Results" slider (1-20)
4. Click "🔍 Search Similar Images"
5. View gallery with similarity scores

### Programmatic API
```python
import torch
import pickle
import numpy as np

# Load model and database
device = torch.device("mps")  # or "cuda" or "cpu"
model = torch.hub.load("./dinov3", "dinov3_vits16", source="local")
model = model.to(device).eval()

with open("embeddings_database.pkl", "rb") as f:
    db = pickle.load(f)

# Extract query embedding
def get_embedding(image):
    # [preprocessing code here]
    with torch.no_grad():
        emb = model(image_tensor)
    return emb.cpu().numpy().flatten()

# Search
query_emb = get_embedding(query_image)
query_emb /= np.linalg.norm(query_emb)  # Normalize
similarities = db['embeddings'] @ query_emb
top_5 = np.argsort(similarities)[::-1][:5]
results = [db['image_paths'][i] for i in top_5]
```

---

## 📁 Project Structure

```
dinov3_test/
├── README.md                        ← You are here
├── DOCUMENTATION_INDEX.md           ← Documentation navigation
├── QUICK_START.md                   ← Quick reference
├── README_IMAGE_SEARCH.md           ← Complete guide
├── ARCHITECTURE.md                  ← Visual diagrams
│
├── dinov3/                          ← DINOv3 model repository
│
├── download_dataset.py              ← Download CIFAR-10
├── download_mapillary_fukuoka.py    ← Download Mapillary images
├── build_embedding_database.py      ← Extract embeddings
├── image_search_app.py              ← CIFAR-10 web UI
├── mapillary_search_app.py          ← Fukuoka Mapillary web UI
│
├── sample_images/                   ← CIFAR-10 dataset
├── fukuoka_images/                  ← Mapillary dataset
├── embeddings_database.pkl          ← CIFAR-10 embeddings
└── fukuoka_embeddings.pkl           ← Fukuoka embeddings
```

---

## 🌍 Mapillary Integration

### Getting an API Token
1. Sign up at https://www.mapillary.com
2. Visit https://www.mapillary.com/dashboard/developers
3. Click "Register Application"
4. Copy your token (format: `MLY|####|####`)

### Download Images
```bash
python download_mapillary_fukuoka.py \
  --token "MLY|your_token_here" \
  --count 500 \
  --output-dir ./fukuoka_images
```

### Geographic Coverage
- **Current**: Fukuoka, Japan (33.59°N, 130.40°E)
- **Customizable**: Modify `fukuoka_center` in script
- **Grid-based**: Searches ~0.8km tiles to cover city

---

## 🔧 Advanced Usage

### Use Different DINOv3 Models
```bash
# Larger model (better accuracy, slower)
python build_embedding_database.py --model dinov3_vitb16

# Launch with larger model
python image_search_app.py --model dinov3_vitb16
```

Available models:
- `dinov3_vits16` (21M params, 384-dim) ← **Currently used**
- `dinov3_vitb16` (86M params, 768-dim)
- `dinov3_vitl16` (304M params, 1024-dim)

### Custom Datasets
```bash
# 1. Organize images
your_dataset/
  ├── category1/
  └── category2/

# 2. Build embeddings
python build_embedding_database.py \
  --images-dir ./your_dataset \
  --output ./your_embeddings.pkl

# 3. Launch UI
python image_search_app.py --database ./your_embeddings.pkl
```

### Scale to Larger Datasets
For 10,000+ images, consider:
- **FAISS**: Approximate nearest neighbor search
- **Database sharding**: Geographic/categorical partitioning
- **Dimensionality reduction**: PCA to reduce storage

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Use different port
python image_search_app.py --port 7862
```

### Out of Memory
- Close other applications
- Use smaller model: `dinov3_vits16`
- Process images one at a time

### Slow Performance
- Check GPU is detected: `torch.backends.mps.is_available()`
- Verify embeddings are pre-computed
- Update PyTorch to latest version

### Poor Search Results
- Ensure query image similar to database content
- Check similarity scores (low scores = dissimilar)
- Verify preprocessing applied correctly

**See [QUICK_START.md](QUICK_START.md) and [README_IMAGE_SEARCH.md](README_IMAGE_SEARCH.md) for detailed troubleshooting.**

---

## 📈 Performance Metrics

### Embedding Extraction
- Apple Silicon (MPS): ~55-77 images/second
- CUDA GPU: ~80-120 images/second
- CPU: ~5-10 images/second

### Search Speed
- 1,000 images: < 1ms
- 10,000 images: ~10ms
- 100,000 images: ~100ms

### Storage Requirements
- Per image: ~1.5 KB (384-dim float32)
- 1,000 images: ~1.5 MB
- 100,000 images: ~150 MB

---

## 🔗 References

### DINOv3
- **Paper**: [DINOv3: Vision Foundation Model (arXiv:2508.10104)](https://arxiv.org/abs/2508.10104)
- **Official Site**: https://ai.meta.com/dinov3/
- **GitHub**: https://github.com/facebookresearch/dinov3

### Mapillary
- **API Documentation**: https://www.mapillary.com/developer/api-documentation
- **Platform**: https://www.mapillary.com/

### Technologies
- **PyTorch**: https://pytorch.org/
- **Gradio**: https://gradio.app/
- **NumPy**: https://numpy.org/

---

## 📝 License & Credits

**DINOv3 Model:**
- Developed by Meta AI Research
- License: Apache 2.0

**Datasets:**
- CIFAR-10: Learning Multiple Layers of Features from Tiny Images (Krizhevsky, 2009)
- Mapillary: Crowdsourced street-level imagery

**This Implementation:**
- Educational and research purposes
- Uses publicly available tools and APIs

---

## 🎓 Learn More

- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation guide
- **[QUICK_START.md](QUICK_START.md)** - Quick commands and reference
- **[README_IMAGE_SEARCH.md](README_IMAGE_SEARCH.md)** - Technical deep dive
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Visual diagrams and architecture

---

**System Version:** 1.0
**Last Updated:** 2025-12-08
**Status:** ✅ Both systems operational
- CIFAR-10 Demo: http://localhost:7860
- Fukuoka Mapillary: http://localhost:7861
