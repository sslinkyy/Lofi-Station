# Quick Start Guide

Get up and running with Image-to-Scene in 15 minutes.

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with 16GB+ VRAM (recommended)
- **CUDA 11.8 or 12.1**
- **Blender 4.0+**
- **~50GB free disk space** (for models)

## Step 1: Set Up AI Worker (5 min)

### Install PyTorch with CUDA

```bash
cd image-to-scene/ai_worker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Download Models (5 min, one-time)

**Option A: Manual Download (Recommended for Windows)**

```bash
# Depth Pro
git clone https://github.com/apple/ml-depth-pro external/depth-pro
cd external/depth-pro
./get_pretrained_models.sh  # Downloads weights
cd ../..

# The following models will auto-download on first use:
# - Qwen3-VL-8B (from Hugging Face)
# - SAM2 (from Meta)
# - OneFormer (from Hugging Face)
```

**Option B: Python Script (Coming Soon)**

```bash
python download_models.py  # Will download all models automatically
```

### Test Worker

```bash
# Start worker
python main.py

# You should see:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Loading AI models...
```

**Test in browser:** http://127.0.0.1:8000/health

Should return:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "0.1.0"
}
```

## Step 2: Install Blender Add-on (2 min)

### Method 1: Development Install (Recommended)

1. Open Blender
2. Edit → Preferences → File Paths
3. Note your "Scripts" path (e.g., `C:\Users\YourName\AppData\Roaming\Blender Foundation\Blender\4.0\scripts`)
4. Navigate to `scripts\addons\`
5. Create symlink or copy `blender_addon` folder:

```bash
# Windows (run as Administrator)
mklink /D "C:\Users\YourName\...\scripts\addons\image_to_scene" "C:\Station\image-to-scene\blender_addon"

# Or just copy the folder
cp -r blender_addon "C:\Users\YourName\...\scripts\addons\image_to_scene"
```

6. Restart Blender
7. Edit → Preferences → Add-ons
8. Search "Image to Scene"
9. Enable the checkbox

### Method 2: ZIP Install

1. Zip the `blender_addon` folder
2. In Blender: Edit → Preferences → Add-ons → Install...
3. Select the zip file
4. Enable "Image to Scene"

### Configure Add-on

1. In Add-on preferences, set:
   - **Worker URL:** `http://127.0.0.1:8000`
2. Click "Test Worker Connection"
3. Should show: ✓ Connected! Status: healthy

## Step 3: First Scene (3 min)

1. Open Blender (new scene)
2. Press `N` → switch to "Image to Scene" tab
3. You should see the panel with "Worker Status" showing your URL
4. Click "Test Connection" → should succeed

**Current Status:** The add-on is installed and can communicate with the worker!

**Next Steps:** Implement the actual pipeline (see IMPLEMENTATION_PLAN.md)

## Troubleshooting

### "Cannot connect to worker"

**Solution:**
```bash
# Make sure worker is running
cd ai_worker
source venv/bin/activate  # or venv\Scripts\activate
python main.py
```

### "CUDA out of memory"

**Solution:**
- Reduce batch size or use smaller models
- Close other GPU applications
- Enable 4-bit quantization in `config/default.yaml`

### "Module not found: transformers"

**Solution:**
```bash
pip install git+https://github.com/huggingface/transformers
```

### Models not loading

**Solution:**
- Check `ai_worker/main.py` - model loading is currently placeholder
- Implementation coming in next phase

## What's Working Now?

✅ **Infrastructure:**
- AI worker runs (FastAPI)
- Blender add-on loads
- HTTP communication works

⏳ **In Progress:**
- Model loading
- Depth estimation
- Scene graph generation
- Blender scene building

📋 **Coming Next:**
- Phase 1: Core pipeline implementation
- Phase 2: Materials & lighting
- Phase 3: Manual refinement UI

## Development Workflow

**Terminal 1: AI Worker**
```bash
cd ai_worker
source venv/bin/activate
python main.py  # Auto-reloads on code changes
```

**Terminal 2: Testing**
```bash
cd tests
pytest -v
```

**Blender:**
- Restart Blender after add-on changes
- Or: In Blender Python console, run:
  ```python
  import bpy
  bpy.ops.preferences.addon_refresh()
  ```

## Next Steps

See [IMPLEMENTATION_PLAN.md](../ROBUST_IMPLEMENTATION_PLAN.md) for the full development roadmap.

**Immediate next tasks:**
1. Implement model loading in `ai_worker/main.py`
2. Create depth estimation service
3. Create VLM scene graph service
4. Build Blender scene builder

**Want to contribute?** Check open issues or start with TODO comments in the code.
