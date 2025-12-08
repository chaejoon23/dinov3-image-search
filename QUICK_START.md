# Quick Start Guide - DINOv3 Image Search

Quick reference for running the DINOv3 image similarity search systems.

---

## 🚀 Launch Web UIs

### CIFAR-10 Demo System
```bash
cd /Users/junchae/dev/dinov3_test
source venv/bin/activate
python image_search_app.py --port 7860
```
**Access:** http://localhost:7860

### Fukuoka Mapillary System
```bash
cd /Users/junchae/dev/dinov3_test
source venv/bin/activate
python mapillary_search_app.py --port 7861
```
**Access:** http://localhost:7861

---

## 📥 Download Data

### CIFAR-10 Dataset (1000 images)
```bash
python download_dataset.py
```
- Output: `./sample_images/`
- 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Mapillary Fukuoka Images
```bash
python download_mapillary_fukuoka.py \
  --token "MLY|your_token_here" \
  --count 500 \
  --output-dir ./fukuoka_images
```
- Get token: https://www.mapillary.com/dashboard/developers
- Output: `./fukuoka_images/` + `metadata.json`

---

## 🧠 Build Embeddings

### CIFAR-10
```bash
python build_embedding_database.py \
  --images-dir ./sample_images \
  --output ./embeddings_database.pkl \
  --model dinov3_vits16
```
- Output: `embeddings_database.pkl` (~1.46 MB for 1000 images)

### Fukuoka Mapillary
```bash
python build_embedding_database.py \
  --images-dir ./fukuoka_images \
  --output ./fukuoka_embeddings.pkl \
  --model dinov3_vits16
```
- Output: `fukuoka_embeddings.pkl` (~0.36 MB for 243 images)

---

## 🔍 How to Search

1. **Open web UI** in browser (see URLs above)
2. **Upload image** - Click upload box, select your image
3. **Adjust results** - Move slider (1-20 results)
4. **Click search** - "🔍 Search Similar Images" button
5. **View results** - Gallery + metadata below

---

## 🎯 What Works Best

### CIFAR-10 Demo
- Upload images of: airplanes, cars, animals, ships, trucks
- Best with clear, centered objects
- 32×32 source resolution (upscaled to 224×224)

### Fukuoka Mapillary
- Upload: street views, urban scenes, buildings, parks
- Works with: architectural photos, outdoor urban environments
- Japanese signage/architecture = stronger matches

---

## 🛠️ Common Commands

### Check if servers are running
```bash
lsof -i :7860  # CIFAR-10
lsof -i :7861  # Fukuoka Mapillary
```

### Stop servers
```bash
# Find process ID (PID)
lsof -i :7860

# Kill process
kill -9 <PID>
```

### Check database info
```python
import pickle
with open('embeddings_database.pkl', 'rb') as f:
    db = pickle.load(f)
print(f"Images: {len(db['image_paths'])}")
print(f"Dimensions: {db['embedding_dim']}")
print(f"Model: {db['model_name']}")
```

### Verify environment
```bash
# Check Python environment
which python  # Should be in venv

# Check packages
pip list | grep -E "torch|gradio|numpy|pillow"

# Check files
ls -lh *.pkl
```

---

## 📊 Understanding Results

### Similarity Scores

| Score | Meaning | Example |
|-------|---------|---------|
| 0.90+ | Nearly identical | Same location, different angle |
| 0.75-0.90 | Very similar | Same object type/category |
| 0.60-0.75 | Moderately similar | Related scenes (both urban/nature) |
| 0.40-0.60 | Somewhat similar | Shared colors/textures |
| < 0.40 | Dissimilar | Different content |

---

## 🗺️ Mapillary Metadata

Each result includes:
- 📅 **Capture date/time** - When photo was taken
- 📍 **GPS coordinates** - Exact location (lat, lon)
- 🧭 **Compass angle** - Camera direction in degrees
- 📐 **Image dimensions** - Width × height in pixels
- 👤 **Photographer** - Mapillary username
- 🔗 **View link** - Open on Mapillary website

---

## ⚙️ System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional (MPS/CUDA) for faster processing
- **Storage**:
  - CIFAR-10: ~100MB images + 1.5MB embeddings
  - Mapillary: ~150MB images + 0.5MB embeddings per 243 images

---

## 🐛 Quick Troubleshooting

### Port already in use
```bash
python image_search_app.py --port 7862  # Use different port
```

### Out of memory
- Close other applications
- Restart Python session
- Use smaller dataset

### Model won't load
```bash
# Verify dinov3 directory exists
ls dinov3/hubconf.py

# Check you're in correct directory
pwd  # Should end in dinov3_test
```

### Slow search
- Embeddings not precomputed → Run `build_embedding_database.py`
- Large database → Normal (search time = O(N))

### No results / low scores
- Query image very different from database content
- Try images similar to dataset (urban for Mapillary, objects for CIFAR-10)

---

## 📚 Full Documentation

See **README_IMAGE_SEARCH.md** for:
- Detailed technical explanations
- DINOv3 architecture deep dive
- Advanced usage and customization
- Integration with your own applications

---

## 🔗 Useful Links

- **Mapillary API Docs**: https://www.mapillary.com/developer/api-documentation
- **DINOv3 Paper**: https://arxiv.org/abs/2508.10104
- **Gradio Docs**: https://gradio.app/docs/
- **PyTorch Hub**: https://pytorch.org/hub/

---

**Need help?** Check the full documentation in `README_IMAGE_SEARCH.md`
