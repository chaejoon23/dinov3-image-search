# System Architecture - DINOv3 Image Search

Visual guide to the system architecture and data flow.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│                                                                     │
│  ┌────────────────────┐              ┌────────────────────┐       │
│  │  CIFAR-10 Demo     │              │  Fukuoka Mapillary │       │
│  │  localhost:7860    │              │  localhost:7861    │       │
│  │                    │              │                    │       │
│  │  • Upload image    │              │  • Upload image    │       │
│  │  • Select top-K    │              │  • Select top-K    │       │
│  │  • View gallery    │              │  • View + metadata │       │
│  └──────────┬─────────┘              └─────────┬──────────┘       │
│             │                                   │                  │
└─────────────┼───────────────────────────────────┼──────────────────┘
              │                                   │
              └──────────┬───────────────────────┘
                         │ HTTP/Gradio
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND SERVICES                               │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              Query Processing Pipeline                     │   │
│  │                                                            │   │
│  │  Upload → Preprocess → DINOv3 → L2 Norm → Search → Rank   │   │
│  │   Image      ↓          Embed.     ↓         ↓      ↓     │   │
│  │           224x224       384-d    Unit Vec  Cosine  Top-K  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    DINOv3 Model                            │   │
│  │                                                            │   │
│  │  Architecture: Vision Transformer (ViT-S/16)              │   │
│  │  Parameters: 21 million                                   │   │
│  │  Input: 224×224 RGB                                       │   │
│  │  Output: 384-dimensional embedding                        │   │
│  │  Device: MPS (Apple Silicon GPU)                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE                                   │
│                                                                     │
│  ┌──────────────────────┐         ┌──────────────────────┐        │
│  │  CIFAR-10 Database   │         │  Fukuoka Database    │        │
│  │                      │         │                      │        │
│  │  • 1000 embeddings   │         │  • 243 embeddings    │        │
│  │  • 384 dimensions    │         │  • 384 dimensions    │        │
│  │  • 1.46 MB size      │         │  • 0.36 MB size      │        │
│  │  • 10 categories     │         │  • GPS metadata      │        │
│  └──────────────────────┘         └──────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete Workflow

### Phase 1: Offline Database Building

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATABASE CONSTRUCTION                        │
└─────────────────────────────────────────────────────────────────┘

┌────────────┐
│  Raw Data  │
│  Source    │
└─────┬──────┘
      │
      ├─────────────────┐
      │                 │
      ▼                 ▼
┌────────────┐    ┌────────────┐
│  CIFAR-10  │    │ Mapillary  │
│  Download  │    │  API Call  │
└─────┬──────┘    └─────┬──────┘
      │                 │
      └────────┬────────┘
               │
               ▼
       ┌──────────────┐
       │  Image Files │
       │   (.jpg)     │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Preprocessing│
       │              │
       │ 1. Resize    │
       │    to 256    │
       │              │
       │ 2. CenterCrop│
       │    to 224    │
       │              │
       │ 3. Normalize │
       │    ImageNet  │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   DINOv3     │
       │   Model      │
       │              │
       │  Input: 224³ │
       │  Output: 384 │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ L2 Normalize │
       │              │
       │ emb = emb/   │
       │   ||emb||    │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Save to      │
       │ Pickle File  │
       │              │
       │ • embeddings │
       │ • paths      │
       │ • metadata   │
       └──────────────┘
```

### Phase 2: Real-Time Search

```
┌─────────────────────────────────────────────────────────────────┐
│                      SIMILARITY SEARCH                          │
└─────────────────────────────────────────────────────────────────┘

     ┌──────────┐
     │  User    │
     │ Uploads  │
     │  Image   │
     └────┬─────┘
          │
          ▼
   ┌─────────────┐
   │ Preprocess  │──────┐  Same preprocessing
   │   Query     │      │  as database images
   └──────┬──────┘      │
          │             │  1. Resize(256)
          │             │  2. CenterCrop(224)
          │             │  3. ToTensor()
          ▼             │  4. Normalize(ImageNet)
   ┌─────────────┐      │
   │   DINOv3    │◄─────┘
   │  Inference  │
   └──────┬──────┘
          │ 384-dim vector
          ▼
   ┌─────────────┐
   │L2 Normalize │
   │  Query Emb  │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────────────────────┐
   │    Cosine Similarity            │
   │                                 │
   │  Load Database Embeddings       │
   │  (N × 384 matrix)               │
   │                                 │
   │  similarities = DB @ query      │
   │                                 │
   │  Returns: N similarity scores   │
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │         Ranking                 │
   │                                 │
   │  1. Sort by score (descending)  │
   │  2. Select top-K                │
   │  3. Fetch image paths           │
   │  4. Load metadata               │
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │      Display Results            │
   │                                 │
   │  • Gallery of K images          │
   │  • Similarity scores            │
   │  • Metadata (GPS, date, etc.)   │
   └─────────────────────────────────┘
```

---

## DINOv3 Model Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    DINOv3 ViT-S/16 Architecture                   │
└───────────────────────────────────────────────────────────────────┘

Input Image (224 × 224 × 3)
      │
      ▼
┌─────────────────────┐
│  Patch Embedding    │  Divide into 16×16 patches
│                     │  224/16 = 14 patches per side
│  14 × 14 = 196     │  = 196 total patches
│    patches          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Add Position       │  Learnable position embeddings
│  Embeddings         │  Help model understand spatial layout
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Transformer Block  │  ┐
│   • Multi-Head      │  │
│     Self-Attention  │  │
│   • Layer Norm      │  │
│   • MLP             │  │  × 12 layers
│   • Residual        │  │  (Small model)
│     Connections     │  │
└──────┬──────────────┘  │
       │                 │
       ▼                 │
       ⋮                 │
       ▼                 ┘
┌─────────────────────┐
│  Final Layer Norm   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Class Token        │  Extract [CLS] token
│  Extraction         │  = Image representation
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  384-dimensional    │  Final embedding vector
│  Feature Vector     │  Captures semantic content
└─────────────────────┘
```

---

## Image Preprocessing Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│              Preprocessing Transformations                     │
└────────────────────────────────────────────────────────────────┘

Original Image (any size)
      │
      ▼
┌─────────────────────┐
│   Resize(256)       │  Shorter side → 256 pixels
│                     │  Maintains aspect ratio
│  Example:           │
│  800×600 → 341×256  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  CenterCrop(224)    │  Extract center 224×224
│                     │
│  ┌─────────────┐   │
│  │             │   │  Focus on main subject
│  │   224×224   │   │  (usually centered)
│  │             │   │
│  └─────────────┘   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   ToTensor()        │  PIL Image → Tensor
│                     │
│  • RGB channels     │  [0,255] int → [0.0,1.0] float
│  • Shape: 3×224×224 │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Normalize()       │  Match training distribution
│                     │
│  For each channel:  │
│  x = (x - μ) / σ    │
│                     │
│  μ = [0.485, 0.456, │  ImageNet statistics
│       0.406]        │
│  σ = [0.229, 0.224, │
│       0.225]        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Batch Dimension    │  Add batch: 1×3×224×224
│  unsqueeze(0)       │  Ready for model input
└─────────────────────┘
```

---

## Similarity Computation Details

```
┌────────────────────────────────────────────────────────────────┐
│                  Cosine Similarity Computation                 │
└────────────────────────────────────────────────────────────────┘

Query Embedding (384-dim)        Database Embeddings (N × 384)
      q                                    D
      │                                    │
      │   Both L2-normalized:             │
      │   ||q|| = 1, ||d_i|| = 1          │
      │                                    │
      └──────────┬───────────────────────┘
                 │
                 ▼
       ┌─────────────────────┐
       │  Matrix-Vector Mult │
       │                     │
       │  similarities =     │
       │    D @ q            │
       │                     │
       │  Shape: (N,)        │
       └──────┬──────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Similarity Scores  │
    │                     │
    │  [0.92, 0.87, ...  │  N values
    │   0.65, 0.43, ...]  │  Range: -1 to 1
    └──────┬──────────────┘   (typically 0 to 1)
           │
           ▼
    ┌─────────────────────┐
    │   Sort Descending   │
    │                     │
    │  indices = argsort  │
    │    (scores)[::-1]   │
    └──────┬──────────────┘
           │
           ▼
    ┌─────────────────────┐
    │   Select Top-K      │
    │                     │
    │  top_k = indices    │
    │    [:k]             │
    └──────┬──────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Return Results     │
    │                     │
    │  images[top_k]      │
    │  scores[top_k]      │
    └─────────────────────┘
```

### Geometric Interpretation

```
          Embedding Space (384-dimensional, visualized in 2D)

                    │ Dimension 2
                    │
         Similar    │    Query
         Image 1    │    Image
            •       │      •
                    │     ╱
         •          │    ╱  θ ≈ 20° → cos(θ) ≈ 0.94
      Similar      │   ╱
      Image 2      │  ╱
                    │ ╱
    ────────────────•──────────────── Dimension 1
                   ╱│
                  ╱ │
                 ╱  │
                ╱   │    θ ≈ 60° → cos(θ) ≈ 0.50
               •    │
          Dissimilar│
           Image    │

    Cosine Similarity = cos(θ)
    • Small angle (< 30°) = High similarity (> 0.85)
    • Medium angle (30-60°) = Moderate similarity (0.5-0.85)
    • Large angle (> 60°) = Low similarity (< 0.5)
```

---

## Database File Structure

### Pickle File Format

```
embeddings_database.pkl
├─ 'embeddings': numpy.ndarray
│  ├─ Shape: (1000, 384)
│  ├─ dtype: float32
│  └─ Memory: ~1.46 MB
│
├─ 'image_paths': list[str]
│  ├─ Length: 1000
│  └─ Example: './sample_images/airplane/airplane_00001.png'
│
├─ 'model_name': str
│  └─ Value: 'dinov3_vits16'
│
└─ 'embedding_dim': int
   └─ Value: 384
```

### Mapillary Metadata JSON

```json
metadata.json (array of objects)
[
  {
    "id": "1000780244899381",                    // Mapillary image ID
    "local_path": "fukuoka_images/...",          // Local file path
    "mapillary_url": "https://...",              // Thumbnail URL
    "captured_at": "2023-04-15T08:30:00Z",      // ISO 8601 timestamp
    "compass_angle": 123.4,                      // Camera direction (degrees)
    "geometry": {
      "type": "Point",
      "coordinates": [130.4017, 33.5904]         // [longitude, latitude]
    },
    "creator": {
      "username": "photographer123"              // Contributor
    },
    "width": 1024,                                // Image width (pixels)
    "height": 768                                 // Image height (pixels)
  },
  ...
]
```

---

## Performance Characteristics

### Time Complexity

```
Operation                    Complexity    Typical Time
─────────────────────────────────────────────────────────
Build Database (N images)    O(N)          ~10-15 ms/image
Query Preprocessing          O(1)          ~50 ms
DINOv3 Inference (1 image)   O(1)          ~10-20 ms (GPU)
Similarity Computation       O(N)          ~1 ms (N=1000)
Total Search Time            O(N)          ~70 ms (N=1000)
```

### Space Complexity

```
Component              Memory Usage
──────────────────────────────────────
Model (ViT-S/16)       ~84 MB (FP32)
Embeddings (N imgs)    N × 384 × 4 bytes
                       = N × 1.5 KB
  • 1,000 images       ~1.5 MB
  • 10,000 images      ~15 MB
  • 100,000 images     ~150 MB
```

---

## Scaling Considerations

### Small Scale (< 10K images)
- ✅ Simple numpy arrays sufficient
- ✅ Linear search acceptable
- ✅ Single machine deployment

### Medium Scale (10K - 1M images)
- ⚠️ Consider approximate nearest neighbors (FAISS)
- ⚠️ Index for sub-linear search time
- ✅ Still single machine feasible

### Large Scale (> 1M images)
- ❗ Requires distributed system
- ❗ Sharding/partitioning needed
- ❗ Consider dimensionality reduction (PCA)
- ❗ Caching layer for hot queries

---

## Data Flow Summary

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│   Images   │────▶│ Preprocess │────▶│  DINOv3    │────▶│ L2 Norm    │
│  (JPEG)    │     │ Pipeline   │     │  Model     │     │ Embeddings │
└────────────┘     └────────────┘     └────────────┘     └─────┬──────┘
                                                                 │
                   224×224×3 RGB      384-dim raw            384-dim
                   Normalized         features              unit vectors
                                                                 │
                                                                 ▼
                                                          ┌────────────┐
                                                          │  Database  │
                                                          │  (Pickle)  │
                                                          └─────┬──────┘
                                                                │
                   ┌────────────────────────────────────────────┘
                   │
                   ▼
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│   Query    │────▶│  Extract   │────▶│  Compute   │────▶│   Rank     │
│   Image    │     │ Embedding  │     │Similarity  │     │  Results   │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
                                                                 │
                   Same pipeline     Matrix × Vector           │
                                     = Score array              ▼
                                                          ┌────────────┐
                                                          │   Display  │
                                                          │    to User │
                                                          └────────────┘
```

---

**For detailed implementation, see README_IMAGE_SEARCH.md**
**For quick commands, see QUICK_START.md**
