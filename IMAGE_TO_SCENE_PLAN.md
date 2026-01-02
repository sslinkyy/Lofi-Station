# Image-to-Blender-Scene Tool: Complete Implementation Plan

## Project Goal
Build a Blender add-on that converts a single image (or video frames) into an accurate, editable 3D scene using AI vision models and VLM-guided reconstruction.

## Core Architecture

### System Components

```
┌─────────────────┐         ┌──────────────────┐
│  Blender Add-on │ ◄─HTTP─► │   AI Worker      │
│  (User UI)      │         │   (FastAPI)      │
└─────────────────┘         └──────────────────┘
        │                            │
        │                            ├─ Depth Anything V2
        │                            ├─ SAM2 Segmentation
        │                            ├─ Qwen2-VL (VLM brain)
        │                            └─ Cache layer
        │
        └─ Scene builder
        └─ Material system
        └─ Verification renderer
```

### Tech Stack

**Blender Add-on (Python 3.11+)**
- Blender 4.0+ API
- Requests (HTTP client)
- JSON schema validation
- UI panels + operators

**AI Worker Service (Separate Python venv)**
- FastAPI (async web framework)
- PyTorch 2.0+
- Transformers (Hugging Face)
- Depth Anything V2
- SAM2
- Qwen2-VL or Qwen2.5-VL
- Pillow, NumPy
- Redis (optional caching)

---

## Data Flow Pipeline

### Single Image Mode (MVP)

```
1. IMAGE INPUT
   └─► Worker: Depth Estimation (DepthAnything V2)
   └─► Worker: Segmentation (SAM2)
   └─► Worker: VLM Scene Graph Planning (Qwen2-VL)
        ├─ Camera estimation
        ├─ Room plane detection
        ├─ Object identification + placement
        ├─ Lighting hypothesis
        └─ Material assignments

2. SCENE GRAPH JSON
   └─► Blender: Build scene from structured data
        ├─ Set camera
        ├─ Create room geometry
        ├─ Place proxy objects
        ├─ Setup materials
        └─ Configure lighting

3. VERIFICATION LOOP
   └─► Blender: Render draft
   └─► Worker: VLM Compare (original vs render)
   └─► Worker: Generate adjustments
   └─► Blender: Apply deltas
   └─► Repeat 2-5 times
```

### Multi-View Mode (Future)

```
1. VIDEO/MULTI-PHOTO INPUT
   └─► Worker: Frame extraction
   └─► Worker: Camera pose estimation (COLMAP)
   └─► Worker: NeRF / Gaussian Splat reconstruction

2. MESH EXPORT + CLEANUP
   └─► Worker: Splat to mesh conversion
   └─► Worker: VLM-guided scene parsing
   └─► Blender: Import + proxy replacement
```

---

## API Contracts

### Worker Service Endpoints

#### `POST /api/analyze`
**Request:**
```json
{
  "image_base64": "...",
  "settings": {
    "mode": "single_image",
    "style_preset": "lofi_cozy",
    "depth_model": "depth_anything_v2",
    "reference_measurement": {
      "px_coords": [[x1,y1], [x2,y2]],
      "real_world_meters": 1.8
    }
  }
}
```

**Response:**
```json
{
  "depth_map_base64": "...",
  "segmentation_masks": {
    "bed": "base64_mask",
    "person": "base64_mask",
    "dog": "base64_mask",
    "window": "base64_mask"
  },
  "scene_graph": {
    "camera": {
      "fov_deg": 42,
      "pitch_deg": -8,
      "yaw_deg": 0,
      "roll_deg": 0,
      "position_hint": [0, -4.2, 1.5]
    },
    "room": {
      "dimensions_m": [3.5, 4.0, 2.6],
      "planes": [
        {"name": "floor", "normal": [0,0,1], "distance": 0.0},
        {"name": "back_wall", "normal": [0,1,0], "distance": 4.0},
        {"name": "right_wall", "normal": [-1,0,0], "distance": 3.5}
      ],
      "window": {
        "wall": "back_wall",
        "rect_world": {"x": 0.6, "y": 1.2, "width": 1.4, "height": 1.8}
      }
    },
    "objects": [
      {
        "id": "bed_01",
        "type": "bed",
        "bbox_px": [90, 520, 1500, 1030],
        "world_position": [1.2, 2.0, 0.4],
        "world_rotation": [0, 0, 15],
        "world_scale": [2.0, 1.6, 0.6],
        "proxy_type": "box_subdiv",
        "depth_order": "mid"
      },
      {
        "id": "person_01",
        "type": "person",
        "bbox_px": [820, 430, 1500, 980],
        "world_position": [1.5, 2.1, 0.9],
        "proxy_type": "card_cutout",
        "depth_order": "front"
      },
      {
        "id": "dog_01",
        "type": "dog",
        "bbox_px": [160, 700, 900, 1040],
        "world_position": [0.6, 2.2, 0.7],
        "proxy_type": "lowpoly_mesh",
        "depth_order": "front"
      }
    ],
    "lighting": {
      "key_light": {
        "type": "area",
        "position": [-0.5, 1.8, 1.6],
        "rotation": [45, -30, 0],
        "temp_kelvin": 2700,
        "strength": 120,
        "size": 0.3
      },
      "fill_light": {
        "type": "area",
        "position": [2.0, 0.5, 2.0],
        "temp_kelvin": 6500,
        "strength": 25,
        "size": 0.8
      },
      "window_emission": {
        "temp_kelvin": 9000,
        "strength": 12
      }
    },
    "materials": [
      {
        "target_id": "bed_01",
        "base_color": [0.82, 0.76, 0.71],
        "roughness": 0.75,
        "metallic": 0.0,
        "texture_hint": "fabric_soft"
      },
      {
        "target_id": "window",
        "shader_type": "emission_glass",
        "emission_strength": 12,
        "base_color": [0.7, 0.85, 1.0]
      }
    ]
  },
  "reasoning": "VLM explanation of decisions made..."
}
```

#### `POST /api/verify`
**Request:**
```json
{
  "original_image_base64": "...",
  "rendered_image_base64": "...",
  "current_scene_graph": { ... },
  "iteration": 2
}
```

**Response:**
```json
{
  "match_score": 0.78,
  "adjustments": [
    {
      "target": "camera",
      "parameter": "fov_deg",
      "current": 42,
      "suggested": 38,
      "reason": "Field of view appears too wide compared to reference"
    },
    {
      "target": "bed_01",
      "parameter": "world_rotation.z",
      "current": 15,
      "suggested": 21,
      "reason": "Bed angle misaligned with perspective lines"
    }
  ],
  "continue": true
}
```

---

## Blender Add-on Structure

```
image_to_scene/
├── __init__.py              # Add-on registration
├── ui/
│   ├── panel.py            # Main UI panel
│   └── operators.py        # Blender operators (buttons)
├── core/
│   ├── worker_client.py    # HTTP client for AI worker
│   ├── scene_builder.py    # Builds Blender scene from JSON
│   ├── camera_solver.py    # Camera setup utilities
│   ├── geometry.py         # Room planes + proxy meshes
│   ├── materials.py        # Material creation
│   ├── lighting.py         # Light setup
│   └── verification.py     # Render + compare loop
├── utils/
│   ├── io.py              # Image load/save, base64
│   ├── cache.py           # Local result caching
│   └── validation.py      # JSON schema validation
└── presets/
    ├── lofi_cozy.json     # Style preset definitions
    └── realistic.json
```

### Key Operators

**`IMAGE_TO_SCENE_OT_analyze`**
- Opens file browser to select image
- Sends to worker `/api/analyze`
- Stores results in scene properties
- Shows preview in UI

**`IMAGE_TO_SCENE_OT_build_scene`**
- Reads scene_graph JSON from properties
- Calls `scene_builder.build_from_graph()`
- Creates collections: Reference / Camera / Room / Objects / Lights

**`IMAGE_TO_SCENE_OT_verify_iterate`**
- Renders current view
- Sends to worker `/api/verify`
- Applies adjustments
- Repeats until convergence or max iterations

**`IMAGE_TO_SCENE_OT_export_pack`**
- Saves .blend file
- Exports textures/maps
- Generates JSON report

---

## AI Worker Implementation

### Directory Structure

```
ai_worker/
├── main.py                  # FastAPI app
├── requirements.txt
├── models/
│   ├── depth.py            # DepthAnything V2 wrapper
│   ├── segmentation.py     # SAM2 wrapper
│   └── vlm.py              # Qwen2-VL wrapper
├── services/
│   ├── analyzer.py         # Orchestrates depth + seg + VLM
│   ├── scene_graph.py      # Scene graph generation logic
│   └── verifier.py         # Render comparison + adjustment
├── prompts/
│   ├── scene_graph.txt     # VLM system prompt for scene analysis
│   └── verify.txt          # VLM prompt for verification
└── cache/
    └── .gitkeep
```

### VLM Prompts

**scene_graph.txt** (excerpt)
```
You are an expert 3D scene reconstruction assistant. Given:
- An input image
- A depth map (grayscale, closer=brighter)
- Segmentation masks for key objects

Your task is to produce a structured JSON scene graph that describes:
1. Camera parameters (FOV, pitch, yaw)
2. Room geometry (planes, dimensions)
3. Object placements (type, position, rotation, scale)
4. Lighting setup (key, fill, ambient)
5. Material assignments

Rules:
- Use depth map to estimate relative distances
- Assume standard room height ~2.6m unless evidence suggests otherwise
- For lofi/cozy style: prioritize simplicity over realism
- Place lights logically (desk lamps emit warm light, windows emit cool)
- Output ONLY valid JSON, no markdown formatting

JSON Schema:
{...}
```

**verify.txt** (excerpt)
```
You are comparing two images:
1. Original reference image
2. Rendered 3D scene attempt

Analyze differences in:
- Camera angle and field of view
- Object positions and rotations
- Lighting direction and intensity
- Overall composition match

Provide specific, measurable adjustments in JSON format.
Only suggest changes that will improve accuracy.
Limit to 3-5 most impactful adjustments per iteration.

Output schema:
{...}
```

---

## Implementation Phases

### Phase 1: Core Pipeline (MVP)
**Goal:** Single image → basic scene that looks right

**Deliverables:**
1. Worker service running locally
   - Depth Anything V2 loaded
   - SAM2 loaded
   - Qwen2-VL loaded
   - `/api/analyze` endpoint working

2. Blender add-on installed
   - UI panel visible
   - Can upload image
   - Receives and displays scene graph JSON

3. Scene builder
   - Camera setup working
   - Room planes created correctly
   - Basic proxy objects placed (boxes for furniture)

4. Materials + lighting (basic)
   - Flat colors from image sampling
   - One key light + one fill

**Success Criteria:**
- Load lofi girl image
- Get back reasonable scene graph
- Build scene in Blender
- Camera view roughly matches original image perspective

**Time Estimate:** 2-3 weeks (assuming models download/run correctly)

---

### Phase 2: Verification Loop
**Goal:** Iterative refinement for accuracy

**Deliverables:**
1. Render capture within Blender
2. `/api/verify` endpoint
3. VLM comparison logic
4. Automatic adjustment application
5. Convergence detection (stop when match_score > 0.85 or iterations > 5)

**Success Criteria:**
- Scene improves visibly after 2-3 iterations
- Final render aligns well with original composition

**Time Estimate:** 1 week

---

### Phase 3: Asset Library + Better Proxies
**Goal:** Replace basic boxes with actual furniture models

**Deliverables:**
1. Asset library system
   - Local folder of .blend files (bed, chair, desk, lamp, etc.)
   - VLM queries library by object type
   - Import into scene programmatically

2. Proxy mesh improvements
   - Subdivided surfaces for organic shapes
   - Card cutouts for characters (with alpha)
   - Better default materials

**Success Criteria:**
- Bed looks like a bed, not a box
- Characters are recognizable

**Time Estimate:** 1-2 weeks

---

### Phase 4: Advanced Materials + Lofi Stylization
**Goal:** Match the aesthetic vibe, not just geometry

**Deliverables:**
1. Texture extraction from image regions
2. Procedural material presets (fabric, wood, glass)
3. Eevee render preset builder:
   - Bloom
   - Depth of field
   - Film grain
   - Color grading nodes

4. "Lofi YouTube" one-click setup

**Success Criteria:**
- Rendered scene has the right mood
- Looks stylistically cohesive

**Time Estimate:** 1 week

---

### Phase 5: Multi-View Mode (Optional)
**Goal:** Accurate geometry from video/multiple photos

**Deliverables:**
1. Video frame extraction
2. Camera pose estimation (COLMAP or OpenCV)
3. NeRF or Gaussian Splat reconstruction
4. Import splat to Blender
5. VLM-guided cleanup (replace splat regions with clean meshes)

**Success Criteria:**
- 10-second phone pan → accurate 3D room
- Editable geometry, not just a splat blob

**Time Estimate:** 3-4 weeks (complex)

---

## Testing Strategy

### Unit Tests
- JSON schema validation
- Depth map processing
- Mask conversion
- Camera math (FOV, position calculation)

### Integration Tests
- Full pipeline: image → scene graph → Blender scene
- Verify loop: iteration count, convergence
- Asset import

### Real-World Test Cases
1. Lofi girl reference image (indoor, stylized)
2. Bedroom photo (standard perspective)
3. Wide-angle living room
4. Outdoor scene (to identify failure modes)

### Performance Benchmarks
- Depth + segmentation: target < 10s on RTX 3060
- VLM scene graph: target < 30s
- Scene build in Blender: target < 5s
- Verification iteration: target < 15s

---

## Development Environment Setup

### AI Worker
```bash
# Create venv
python -m venv ai_worker_env
source ai_worker_env/bin/activate  # or ai_worker_env\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install fastapi uvicorn pillow numpy
pip install depth-anything-v2  # Check actual package name
pip install segment-anything-2  # Check actual package name
pip install qwen-vl-utils  # If available, or use Transformers directly

# Run worker
cd ai_worker
python main.py
```

### Blender Add-on
```bash
# Development installation
# Symlink or copy add-on folder to Blender scripts/addons/
# Enable in Preferences → Add-ons

# Or use Blender's development mode:
blender --python-expr "import sys; sys.path.append('c:/Station/image_to_scene'); import bpy; bpy.ops.preferences.addon_enable(module='image_to_scene')"
```

---

## Configuration Files

### Worker config.yaml
```yaml
models:
  depth:
    name: "depth-anything/Depth-Anything-V2-Large"
    device: "cuda"
  segmentation:
    name: "facebook/sam2-hiera-large"
    device: "cuda"
  vlm:
    name: "Qwen/Qwen2-VL-7B-Instruct"
    device: "cuda"
    quantization: "4bit"  # Optional for lower VRAM

server:
  host: "127.0.0.1"
  port: 8000
  workers: 1

cache:
  enabled: true
  max_size_gb: 10
  ttl_hours: 24
```

### Blender add-on preferences
```python
class ImageToScenePreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    worker_url: bpy.props.StringProperty(
        name="Worker URL",
        default="http://127.0.0.1:8000"
    )

    style_preset: bpy.props.EnumProperty(
        name="Default Style",
        items=[
            ('LOFI', "Lofi Cozy", "Stylized, soft, YouTube lofi aesthetic"),
            ('REALISTIC', "Realistic", "Accurate materials and lighting"),
        ],
        default='LOFI'
    )

    max_verify_iterations: bpy.props.IntProperty(
        name="Max Verification Iterations",
        default=5,
        min=0,
        max=20
    )
```

---

## Error Handling

### Common Failure Modes

1. **Worker offline**
   - Blender add-on: Show clear error message with worker URL
   - Suggest: "Start AI worker with: python ai_worker/main.py"

2. **Out of VRAM**
   - Worker: Catch CUDA OOM, return error response
   - Suggest: Use smaller models or quantization

3. **VLM produces invalid JSON**
   - Worker: Validate with jsonschema before returning
   - Retry with corrective prompt if invalid

4. **Depth map too noisy**
   - Apply bilateral filter before using
   - VLM should handle uncertainty gracefully

5. **No objects detected**
   - Fallback: Create room + image plane in scene
   - Let user manually segment

---

## Future Enhancements (Backlog)

- **User correction tools:** Click to adjust plane positions, object placements
- **Animation support:** Estimate simple loops (breathing, light flicker)
- **Multi-room scenes:** Detect doorways, create connected spaces
- **Outdoor scenes:** Sky dome, terrain estimation
- **Style transfer:** Apply different render styles to same geometry
- **Export to game engines:** GLTF/FBX with baked lighting
- **Collaborative editing:** Multiple users refine same scene
- **Training pipeline:** Fine-tune VLM on synthetic Blender renders for better accuracy

---

## Success Metrics

**MVP Success:**
- 70%+ users can create a recognizable scene from one image in < 5 minutes
- Camera perspective matches reference image within 10° FOV error
- Room dimensions within 20% of actual (if measurable)

**Full Product Success:**
- 90%+ composition accuracy (verified by VLM score)
- Asset-replaced scenes look production-ready for lofi streams
- Multi-view mode produces geometry accurate to < 5cm (with proper calibration)

---

## Next Steps

1. **Set up repository structure**
   - Initialize Git repo
   - Create directory layout
   - Add .gitignore (models/, cache/, venv/)

2. **Start with Phase 1, Milestone 1:**
   - Get worker running with Depth Anything V2
   - Test depth map output quality

3. **Parallel track:**
   - Build Blender add-on UI scaffold
   - Test HTTP communication

4. **Iterate:**
   - Weekly demos of progress
   - Collect real test images
   - Refine VLM prompts based on results

---

## Open Questions to Resolve

1. **VLM model choice:**
   - Qwen2-VL vs Qwen2.5-VL vs other?
   - Test accuracy vs speed tradeoffs

2. **Segmentation strategy:**
   - Fully automatic (SAM2 auto-segment)?
   - Or guided (user clicks object centers)?

3. **Scale calibration:**
   - Always require user to provide one measurement?
   - Or rely on typical object sizes (bed ≈ 2m)?

4. **Multi-view reconstruction tool:**
   - Use existing Gaussian splat add-on or build custom?

5. **Deployment:**
   - Local-only tool?
   - Or cloud worker option for users without GPU?

---

**This plan is ready to execute. Let me know which component you want to build first, or if you want me to start scaffolding the directory structure and code stubs.**
