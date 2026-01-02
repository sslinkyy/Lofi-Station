# Testing Guide - Depth Estimation

## What's Working Now

✅ **AI Worker:**
- FastAPI server runs
- Depth Pro model loading
- Depth estimation from images
- Depth map encoding and caching
- Point cloud generation

## Prerequisites

Before testing, install Depth Pro:

```bash
cd c:\Station\image-to-scene

# Clone Depth Pro
git clone https://github.com/apple/ml-depth-pro external/depth-pro

# Install it
cd external/depth-pro
pip install -e .

# Download weights
./get_pretrained_models.sh  # On Windows, run in Git Bash or WSL
```

**Or download weights manually:**
```bash
# From: https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
# Save to: external/depth-pro/checkpoints/depth_pro.pt
```

## Test 1: Standalone Depth Test (No Server)

Test depth estimation directly without running the server:

```bash
cd c:\Station\image-to-scene\ai_worker

# Install test dependencies
pip install matplotlib scipy

# Test on an image
python test_depth.py path/to/your/image.jpg
```

**Expected output:**
```
============================================================
Testing Depth Estimation
============================================================

Device: cuda
GPU: NVIDIA GeForce RTX 4080

Loading image: test.jpg
Image size: (1920, 1080)

Loading Depth Pro model...
✓ Model loaded

Estimating depth...

✓ Depth estimated:
  Range: 0.52 - 8.34 m
  Mean: 3.12 m
  Cached: False

Visualizing results...
✓ Results saved to: test_outputs/depth_test_test.png
✓ Point cloud generated: 2073600 points
✓ Point cloud saved to: test_outputs/pointcloud_test.ply

============================================================
Test Complete!
============================================================
```

**Output files:**
- `test_outputs/depth_test_*.png` - Visualization (image, depth, confidence)
- `test_outputs/pointcloud_*.ply` - 3D point cloud (open in MeshLab/CloudCompare)

## Test 2: Full Server Test

Test through the FastAPI server:

### Start Server

```bash
cd c:\Station\image-to-scene\ai_worker

# Create venv if not done
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```

**Expected startup:**
```
============================================================
Loading AI models...
============================================================
Using device: cuda
GPU: NVIDIA GeForce RTX 4080
VRAM: 16.0 GB

[1/3] Loading Depth Model...
INFO:depth_pro.depth_pro:Loading depth-pro model...
✓ Depth Pro loaded successfully

[2/3] Loading VLM...
⚠️  VLM loading not yet implemented

[3/3] Loading Segmentation...
⚠️  Segmentation loading not yet implemented

============================================================
✓ All models loaded successfully!
============================================================
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Test Health Endpoint

**Browser:** http://127.0.0.1:8000/health

**Expected:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "0.1.0"
}
```

### Test Depth Analysis (Python)

```python
import requests

# Upload image
with open('test_image.jpg', 'rb') as f:
    files = {'image': f}
    settings = {'settings': '{"mode": "fast", "style_preset": "lofi"}'}

    response = requests.post(
        'http://127.0.0.1:8000/api/analyze',
        files=files,
        data=settings,
        timeout=60
    )

result = response.json()

print(f"Status: {result['status']}")
print(f"Processing time: {result['processing_time_s']:.2f}s")
print(f"Models: {result['model_info']}")

# Depth map is in base64
if result['depth_map_base64']:
    import base64
    from PIL import Image
    from io import BytesIO

    depth_data = base64.b64decode(result['depth_map_base64'])
    depth_image = Image.open(BytesIO(depth_data))
    depth_image.save('depth_output.png')
    print("✓ Depth map saved to depth_output.png")
```

**Expected output:**
```
Status: depth_only
Processing time: 1.23s
Models: {'depth': 'depth-pro (loaded)', 'vlm': 'not loaded', 'segmentation': 'not loaded'}
✓ Depth map saved to depth_output.png
```

### Test via cURL

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "image=@test_image.jpg" \
  -F 'settings={"mode":"fast"}' \
  > response.json
```

## Test 3: Blender Add-on Test

1. **Start server** (as above)

2. **Open Blender**

3. **Install add-on:**
   - Edit → Preferences → Add-ons → Install
   - Select `blender_addon` folder
   - Enable "Image to Scene"

4. **Test connection:**
   - Press N → switch to "Image to Scene" tab
   - Click "Test Worker Connection"

**Expected:**
```
✓ Connected! Status: healthy
```

**Current limitation:** The "Analyze Image" button isn't implemented yet. That's next!

## Troubleshooting

### "ModuleNotFoundError: No module named 'depth_pro'"

**Solution:**
```bash
git clone https://github.com/apple/ml-depth-pro external/depth-pro
cd external/depth-pro
pip install -e .
```

### "FileNotFoundError: checkpoints/depth_pro.pt"

**Solution:**
```bash
cd external/depth-pro
./get_pretrained_models.sh

# Or download manually:
# https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
# Save to: external/depth-pro/checkpoints/depth_pro.pt
```

### "CUDA out of memory"

**Solutions:**
1. Close other applications using GPU
2. Use CPU mode (edit `main.py`: device = "cpu")
3. Reduce image resolution before processing

### "RuntimeError: depth_pro not installed"

Depth Pro isn't in the Python path. Either:
1. Install it: `cd external/depth-pro && pip install -e .`
2. Add to path manually in `ai_worker/main.py`:
   ```python
   import sys
   sys.path.insert(0, 'external/depth-pro')
   ```

## Next Steps

Once depth estimation is working:

1. ✅ **Depth working** ← You are here
2. ⏳ **Add VLM (Qwen3-VL)** - Scene understanding
3. ⏳ **Add Segmentation (SAM2)** - Object masks
4. ⏳ **Scene Graph Generation** - Structured output
5. ⏳ **Blender Scene Builder** - Actually build the scene

See [ROBUST_IMPLEMENTATION_PLAN.md](ROBUST_IMPLEMENTATION_PLAN.md) for full roadmap.

## Performance Benchmarks

Expected performance on RTX 4080:

| Image Size | First Run | Cached |
|------------|-----------|--------|
| 1920x1080 | ~1.2s | ~0.1s |
| 1280x720 | ~0.8s | ~0.05s |
| 3840x2160 | ~3.5s | ~0.2s |

CPU mode is ~10-20x slower.

## Success Criteria

✅ **You've succeeded if:**
1. Server starts without errors
2. `/health` returns `models_loaded: true`
3. Test script generates depth map visualization
4. Server API returns depth map in < 5 seconds
5. Blender add-on can connect to server

**All working?** → Ready for next phase: VLM integration!
