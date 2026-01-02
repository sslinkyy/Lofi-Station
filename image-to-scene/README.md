# Image-to-Blender-Scene

Transform a single image into a fully-realized Blender 3D scene with AI-powered reconstruction.

## Features

- **Fast 3D Reconstruction**: Depth-based or Gaussian Splat reconstruction (< 5s)
- **AI Scene Understanding**: VLM-powered scene graph generation
- **Smart Material Estimation**: Intrinsic image decomposition for realistic materials
- **Intelligent Lighting**: Multi-method lighting estimation
- **Lofi Rendering Presets**: One-click stylized render setups
- **Manual Refinement UI**: Viewport gizmos and sliders for fine-tuning
- **Multi-View Support**: Optional accurate geometry from multiple images

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Blender Add-on │ ◄─HTTP─► │   AI Worker      │
│  (User UI)      │         │   (FastAPI)      │
└─────────────────┘         └──────────────────┘
                                     │
                                     ├─ Depth Pro / SHARP
                                     ├─ SAM2 + OneFormer
                                     ├─ Qwen3-VL
                                     └─ RGB↔X Materials
```

## Quick Start

### 1. Set up AI Worker

```bash
cd ai_worker
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download models (first run)
python download_models.py

# Start worker
python main.py
```

Worker runs at `http://localhost:8000`

### 2. Install Blender Add-on

1. Open Blender → Edit → Preferences → Add-ons
2. Click "Install..."
3. Select `blender_addon` folder
4. Enable "Image to Scene"
5. Configure worker URL in add-on preferences

### 3. Use

1. In Blender, press N → Image to Scene panel
2. Click "Select Image"
3. Provide scale measurement (click two points, enter distance)
4. Click "Analyze Image"
5. Wait 5-20 seconds
6. Scene builds automatically!
7. Refine with viewport gizmos and sliders

## Requirements

### AI Worker
- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (recommended)
- CUDA 11.8 or 12.1
- ~50GB disk space for models

### Blender Add-on
- Blender 4.0+
- Network access to AI worker (localhost or remote)

## Project Structure

```
image-to-scene/
├── ai_worker/              # FastAPI worker service
│   ├── models/            # Model wrappers (depth, VLM, segmentation)
│   ├── services/          # Pipeline orchestration
│   ├── prompts/           # VLM prompt templates
│   └── main.py            # FastAPI app
│
├── blender_addon/         # Blender add-on
│   ├── ui/               # Panels and operators
│   ├── core/             # Scene builder, materials, lighting
│   └── utils/            # HTTP client, validation
│
├── config/                # Configuration files
├── assets/                # Asset library (furniture, etc.)
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Development Roadmap

- [x] Architecture design
- [x] Tool research and verification
- [ ] Phase 0.5: Tool compatibility testing
- [ ] Phase 1: Core pipeline (depth + VLM + scene building)
- [ ] Phase 2: Materials and lighting
- [ ] Phase 3: Manual refinement UI
- [ ] Phase 4: Verification loop
- [ ] Phase 5: Multi-view mode

## License

MIT (code)
Individual model licenses apply (see LICENSES.md)

## Credits

Built using:
- [Apple Depth Pro](https://github.com/apple/ml-depth-pro)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [OneFormer](https://github.com/SHI-Labs/OneFormer)
- [RGB↔X](https://github.com/zheng95z/rgbx)
- [MILo](https://github.com/Anttwo/MILo)

See [CUTTING_EDGE_TOOLS_2025_2026.md](../CUTTING_EDGE_TOOLS_2025_2026.md) for full tool list.
