# Image-to-Scene Plan Refinements

Based on user preferences, here are the key refinements to the original plan.

---

## Core Philosophy Changes

**Original:** Opinionated tool with fixed workflow
**Refined:** Flexible, configurable tool where users choose their preferences

This means more UI configuration, but better adaptability to different use cases (quick lofi scenes vs. accurate architectural reconstruction).

---

## 1. Deployment Architecture (Hybrid Local + Cloud)

### Local Worker Mode
- Runs on user's machine
- Requires 16GB+ VRAM (RTX 4080 / A4000 tier)
- Full privacy, no API costs
- Models loaded on startup:
  - Depth Anything V2 (Large variant)
  - SAM2 (Hiera-Large)
  - Qwen2-VL-7B (4-bit quantization, ~6GB VRAM)
  - Total VRAM usage: ~14-15GB

### Cloud Worker Mode (Fallback/Alternative)
- API-based service (FastAPI hosted on cloud)
- Uses OpenAI GPT-4o for VLM tasks
- Depth/segmentation can still run locally OR cloud
- User provides API key in add-on preferences

### Implementation Strategy
```python
# Blender add-on preferences
class WorkerMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"  # Try local, fallback to cloud

class ImageToScenePreferences(bpy.types.AddonPreferences):
    worker_mode: bpy.props.EnumProperty(
        name="Worker Mode",
        items=[
            ('LOCAL', "Local (GPU Required)", "Run models on your GPU"),
            ('CLOUD', "Cloud (API Required)", "Use cloud service"),
            ('AUTO', "Auto (Hybrid)", "Try local first, fallback to cloud"),
        ],
        default='AUTO'
    )

    local_worker_url: bpy.props.StringProperty(
        name="Local Worker URL",
        default="http://127.0.0.1:8000"
    )

    cloud_api_key: bpy.props.StringProperty(
        name="Cloud API Key",
        subtype='PASSWORD'
    )

    vlm_backend: bpy.props.EnumProperty(
        name="VLM Model",
        items=[
            ('QWEN2VL', "Qwen2-VL-7B (Local)", "Runs on your GPU"),
            ('QWEN25VL', "Qwen2.5-VL-7B (Local)", "Newer, experimental"),
            ('GPT4O', "GPT-4o (Cloud)", "Best quality, requires API key"),
            ('CLAUDE_SONNET', "Claude Sonnet 4.5 (Cloud)", "Excellent vision, requires API key"),
        ],
        default='QWEN2VL'
    )
```

### Worker Service Health Check
```python
# On add-on enable/operator start
def check_worker_health():
    try:
        # Try local
        response = requests.get(f"{local_url}/health", timeout=2)
        if response.ok:
            return WorkerMode.LOCAL
    except:
        pass

    # Try cloud
    if cloud_api_key:
        return WorkerMode.CLOUD

    # Fail gracefully
    show_error("No worker available. Start local worker or configure cloud API.")
    return None
```

---

## 2. User Interaction Model (Semi-Automatic + Refinement)

### Phase A: Automatic Analysis
1. User selects image
2. Tool runs analysis (depth, segmentation, VLM scene graph)
3. Progress bar shows: "Analyzing depth... Segmenting objects... Planning scene..."

### Phase B: Clarification (When Needed)
VLM can request user input during analysis via special JSON response:

```json
{
  "status": "needs_clarification",
  "questions": [
    {
      "type": "label_confirm",
      "message": "Is this a window or a painting?",
      "region_mask_id": "region_5",
      "options": ["Window", "Painting", "Mirror", "Other"]
    },
    {
      "type": "measurement",
      "message": "Provide a real-world measurement for scale accuracy",
      "instruction": "Click two points on the image and enter the distance",
      "optional": true
    }
  ]
}
```

Blender add-on shows modal dialog:
- Displays image with highlighted region
- Radio buttons for quick answer
- "Skip" button if user doesn't know
- Visual markup mode: click to draw correction

### Phase C: Scene Generation
Build scene with user's answers incorporated.

### Phase D: Manual Refinement UI
After scene is built, add custom Blender UI:

#### 3D Viewport Gizmos
```python
# Custom gizmo group for room plane adjustment
class RoomPlaneGizmos(bpy.types.GizmoGroup):
    """
    Visual handles in viewport:
    - Arrows to move wall planes
    - Rotation handles for floor tilt
    - Corner handles to resize room dimensions
    """
    pass

# Custom gizmo for object placement
class ObjectProxyGizmo(bpy.types.GizmoGroup):
    """
    Standard transform gizmo + additional:
    - "Snap to floor" button
    - "Match perspective" mode (align to vanishing points)
    - Depth slider (move along camera ray)
    """
    pass
```

#### Side Panel (N-Panel)
```
┌─────────────────────────────────┐
│ Image to Scene                  │
├─────────────────────────────────┤
│ Scene Controls:                 │
│  [Re-analyze Image]             │
│  [Re-run Verification]          │
│                                 │
│ Camera:                         │
│  FOV: [38°] ←────────→          │
│  Pitch: [-8°] ←────────→        │
│                                 │
│ Lighting:                       │
│  Key Strength: [120] ←────→     │
│  Key Temp: [2700K] ←────→       │
│  Fill Strength: [25] ←────→     │
│  Window Glow: [12] ←────→       │
│                                 │
│ Materials:                      │
│  Global Roughness: [0.75] ←──→  │
│  Color Saturation: [0.8] ←───→  │
│                                 │
│ Selected Object: "bed_01"       │
│  Type: bed                      │
│  Proxy: [box_subdiv ▼]          │
│  Replace with Asset: [Browse]   │
│  Certainty: 85% ████░           │
│                                 │
│ [Apply Changes] [Reset to Auto] │
└─────────────────────────────────┘
```

#### Live Preview
- Changes to sliders update immediately in Eevee viewport
- "Apply Changes" locks in values and optionally re-runs verification

---

## 3. Uncertainty Handling (Multi-Method Clarification)

### Detection
VLM outputs confidence scores per decision:
```json
{
  "objects": [
    {
      "id": "obj_3",
      "type": "lamp",
      "confidence": 0.92,  // High confidence, no clarification needed
      "bbox_px": [...]
    },
    {
      "id": "obj_7",
      "type": "window_or_painting",  // Uncertain
      "confidence": 0.58,
      "bbox_px": [...],
      "needs_clarification": true,
      "clarification_options": ["window", "painting", "mirror"]
    }
  ]
}
```

### Clarification Methods (User Choice)

#### Method 1: Quick Questions (Fastest)
Modal popup with buttons:
```
┌────────────────────────────────┐
│ Help me identify this region:  │
│ [Shows highlighted region]     │
│                                │
│ What is this?                  │
│  ( ) Window                    │
│  ( ) Painting                  │
│  ( ) Mirror                    │
│  ( ) Other: [______]           │
│                                │
│  [Confirm]  [Skip]             │
└────────────────────────────────┘
```

#### Method 2: Visual Markup (Most Precise)
Interactive image view:
- SAM2 masks overlaid with labels
- Click region to correct label
- Draw scribbles to add missing objects
- Click-drag to adjust bbox

```python
class VisualMarkupOperator(bpy.types.Operator):
    """
    Full-screen image editor:
    - Shows original image
    - Overlays segmentation masks (color-coded)
    - Click mask to change label (dropdown appears)
    - Right-click to add new object (SAM2 auto-segments)
    - 'A' to approve, 'R' to re-run segmentation
    """
    pass
```

#### Method 3: Reference Measurements (Most Accurate Scale)
User clicks two points, enters real distance:
```
┌────────────────────────────────┐
│ Click two points on the image: │
│ [Shows image with crosshair]   │
│                                │
│ Point 1: (320, 450) ✓          │
│ Point 2: (890, 451) ✓          │
│                                │
│ Real-world distance:           │
│  [___60___] inches             │
│  Unit: [inches ▼]              │
│                                │
│ This will calibrate scene scale│
│  [Confirm]  [Cancel]           │
└────────────────────────────────┘
```

### Add-on Setting
```python
uncertainty_handling: bpy.props.EnumProperty(
    name="When Uncertain",
    items=[
        ('AUTO', "Best Guess (mark uncertain)", "Fastest"),
        ('QUESTIONS', "Ask Quick Questions", "Balanced"),
        ('MARKUP', "Visual Markup Mode", "Most precise"),
        ('MEASUREMENTS', "Request Measurements", "Best accuracy"),
    ],
    default='QUESTIONS'
)
```

---

## 4. MVP Priority: Visual Quality Over Accuracy

This shifts Phase 1 focus:

### Original Phase 1
- Camera match + room planes + basic proxies
- Depth mesh
- Success: "composition roughly matches"

### Refined Phase 1
- Camera match (good enough, not perfect)
- Room planes (approximate)
- **High-quality materials** (texture sampling, procedural detail)
- **Proper lighting setup** (key + fill + ambient)
- **Eevee render preset** (bloom, DOF, grain)
- **Lofi stylization** (color grading, vignette)
- Success: "render looks professional and matches the mood"

### What This Means for Development Priority

**Reduce scope:**
- Skip verification loop in Phase 1 (move to Phase 2)
- Room planes can be approximate (user can adjust with gizmos)
- Object placement can be rough

**Increase focus:**
- Material system (Phase 4 → Phase 1)
- Lighting estimation (Phase 4 → Phase 1)
- Render compositing (Phase 4 → Phase 1)

**New Phase 1 Checklist:**
1. Depth + segmentation working
2. VLM scene graph (camera, room, objects, **materials**, **lighting**)
3. Scene builder creates geometry
4. **Material assignment with texture extraction**
5. **Light setup (3-point lighting)**
6. **Eevee compositor nodes (bloom, grain, vignette)**
7. One-click "Render Like Reference" button

**Deliverable:**
User uploads lofi girl image, gets back a Blender scene that *renders* like lofi aesthetic immediately, even if geometry is approximate.

---

## 5. VLM Model Selection (User Configurable)

### Supported Models

| Model | Mode | VRAM | Speed | Quality | Cost |
|-------|------|------|-------|---------|------|
| Qwen2-VL-7B | Local | ~6GB (4-bit) | Fast | Good | Free |
| Qwen2.5-VL-7B | Local | ~6GB (4-bit) | Fast | Better | Free |
| LLaVA-NeXT-34B | Local | ~14GB (4-bit) | Slow | Best (local) | Free |
| GPT-4o | Cloud | N/A | Medium | Excellent | ~$0.10/image |
| Claude Sonnet 4.5 | Cloud | N/A | Fast | Excellent | ~$0.08/image |

### Model Registry Pattern
```python
# ai_worker/models/vlm_registry.py
class VLMModel(ABC):
    @abstractmethod
    def generate_scene_graph(self, image, depth_map, masks, prompt):
        pass

class Qwen2VLModel(VLMModel):
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            device_map="auto",
            load_in_4bit=True
        )

    def generate_scene_graph(self, image, depth_map, masks, prompt):
        # Implementation
        pass

class GPT4oModel(VLMModel):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_scene_graph(self, image, depth_map, masks, prompt):
        # Encode images to base64
        # Call OpenAI API with vision
        pass

# Registry
VLM_REGISTRY = {
    "qwen2vl": Qwen2VLModel,
    "qwen25vl": Qwen25VLModel,
    "gpt4o": GPT4oModel,
    "claude_sonnet": ClaudeSonnetModel,
}

def get_vlm(model_name: str, config: dict) -> VLMModel:
    return VLM_REGISTRY[model_name](**config)
```

### Prompt Templates Per Model
Different models need different prompt formats:

```python
# ai_worker/prompts/qwen2vl_scene_graph.txt
QWEN2VL_PROMPT = """
<|im_start|>system
You are an expert 3D scene reconstruction assistant specializing in creating accurate scene graphs from images.
<|im_end|>
<|im_start|>user
Analyze this image and create a JSON scene graph.
Input image: <image>
Depth map: <image>
Requirements: {requirements}
<|im_end|>
<|im_start|>assistant
"""

# ai_worker/prompts/gpt4o_scene_graph.txt
GPT4O_PROMPT = """
You are an expert 3D scene reconstruction assistant.

I will provide:
1. An input image
2. A depth map (grayscale, closer=brighter)
3. Segmentation masks

Your task: Generate a JSON scene graph following this exact schema:
{schema}

Rules:
- Output ONLY valid JSON, no markdown code blocks
- Use depth map for distance estimation
- ...
"""
```

---

## 6. Asset Library (Multi-Format Support)

### Asset Hierarchy
```
assets/
├── blend/                   # Native Blender files (highest priority)
│   ├── furniture/
│   │   ├── bed_modern.blend
│   │   ├── bed_vintage.blend
│   │   ├── desk_simple.blend
│   │   └── chair_office.blend
│   ├── lighting/
│   │   ├── lamp_desk.blend
│   │   └── lamp_floor.blend
│   └── decor/
│       ├── plant_small.blend
│       └── books_stack.blend
│
├── gltf/                    # GLTF imports (fallback)
│   ├── furniture/
│   │   └── bed_basic.glb
│   └── ...
│
├── procedural/              # Geometry node presets (last resort)
│   ├── bed_procedural.blend  # Contains geometry node setup
│   └── ...
│
└── catalog.json             # Asset metadata
```

### Asset Catalog Schema
```json
{
  "assets": [
    {
      "id": "bed_modern_01",
      "type": "bed",
      "tags": ["modern", "minimalist", "lofi"],
      "format": "blend",
      "path": "blend/furniture/bed_modern.blend",
      "object_name": "Bed",  // Name inside .blend file
      "dimensions": [2.0, 1.6, 0.6],  // Default size (m)
      "thumbnail": "thumbnails/bed_modern_01.png",
      "style_match": {
        "lofi": 0.95,
        "realistic": 0.60
      }
    },
    {
      "id": "bed_procedural",
      "type": "bed",
      "tags": ["procedural", "fallback"],
      "format": "procedural",
      "path": "procedural/bed_procedural.blend",
      "node_group_name": "BedGenerator",
      "parameters": {
        "width": 2.0,
        "length": 1.6,
        "height": 0.6,
        "style": "modern"
      }
    }
  ]
}
```

### Asset Selection Logic
```python
def get_best_asset(object_type: str, style_preset: str, vlm_tags: list):
    # 1. Filter by type
    candidates = [a for a in catalog if a["type"] == object_type]

    # 2. Score by style match
    scored = [
        (asset, asset["style_match"].get(style_preset, 0.5))
        for asset in candidates
    ]

    # 3. Boost if VLM tags match
    for asset, score in scored:
        tag_match = len(set(asset["tags"]) & set(vlm_tags))
        score += tag_match * 0.1

    # 4. Prefer blend > gltf > procedural
    format_priority = {"blend": 2, "gltf": 1, "procedural": 0}
    scored = [
        (asset, score + format_priority[asset["format"]])
        for asset, score in scored
    ]

    # 5. Return best match
    best = max(scored, key=lambda x: x[1])
    return best[0]
```

### Procedural Fallback Example
```python
# For missing assets, generate on-the-fly
def create_procedural_bed(width, length, height):
    """
    Uses Geometry Nodes to create a simple bed:
    - Mattress (subdiv cube)
    - Pillows (inflated cubes)
    - Frame (thin boxes)
    """
    bpy.ops.mesh.primitive_cube_add()
    bed = bpy.context.active_object

    # Add geometry nodes modifier
    mod = bed.modifiers.new("BedGenerator", 'NODES')
    node_group = bpy.data.node_groups.get("BedGenerator")

    if not node_group:
        # Create geometry node setup
        node_group = create_bed_geometry_nodes()

    mod.node_group = node_group
    mod["Input_2_attribute_name"] = width
    mod["Input_3_attribute_name"] = length
    mod["Input_4_attribute_name"] = height

    return bed
```

---

## 7. Verification Adjustments (User-Configurable Behavior)

### Adjustment Modes

```python
class AdjustmentMode(Enum):
    AUTOMATIC = "automatic"          # Apply immediately, allow undo
    PREVIEW = "preview"              # Show diff, user approves
    VISUAL_DIFF = "visual_diff"      # Overlay comparison, click to fix
    MANUAL_ONLY = "manual_only"      # Suggestions only, no auto-apply

class ImageToScenePreferences:
    adjustment_mode: bpy.props.EnumProperty(
        name="Verification Adjustments",
        items=[
            ('AUTOMATIC', "Auto-Apply (Recommended)", "Fast iteration, undo available"),
            ('PREVIEW', "Preview First", "Review before applying"),
            ('VISUAL_DIFF', "Visual Diff Overlay", "Most intuitive"),
            ('MANUAL_ONLY', "Suggestions Only", "Full manual control"),
        ],
        default='AUTOMATIC'
    )
```

### Implementation: Automatic Mode
```python
def apply_adjustments(scene_graph, adjustments):
    # Store state for undo
    undo_state = serialize_scene_state()

    for adj in adjustments:
        if adj["target"] == "camera":
            cam = bpy.data.objects["Camera"]
            cam.data.lens = fov_to_focal_length(adj["suggested"])

        elif adj["target"].startswith("obj_"):
            obj = bpy.data.objects[adj["target"]]
            if adj["parameter"] == "world_rotation.z":
                obj.rotation_euler.z = radians(adj["suggested"])

        # Log change
        print(f"Applied: {adj['reason']}")

    # Enable undo
    bpy.ops.ed.undo_push(message="Verification Adjustments Applied")
```

### Implementation: Visual Diff Mode
```python
class VisualDiffOperator(bpy.types.Operator):
    """
    Fullscreen overlay showing:
    - Left: Original image
    - Right: Current render
    - Middle: Overlay blend (slider)
    - Highlighted regions: Areas with high difference

    User interactions:
    - Click difference region → Auto-suggest fix
    - Accept → Apply that adjustment
    - Reject → Mark as "do not adjust"
    """

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # User clicked on a difference region
            region_id = self.get_region_at_mouse(event.mouse_x, event.mouse_y)

            # Find relevant adjustment
            adj = [a for a in self.adjustments if a["affects_region"] == region_id]

            if adj:
                self.show_adjustment_popup(adj[0])

        return {'RUNNING_MODAL'}
```

---

## 8. Animation Options (Configurable Per Scene)

### Animation Presets

```python
class AnimationPreset:
    NONE = "none"
    MINIMAL = "minimal"      # Light flicker + subtle breathing
    LOFI = "lofi"           # Above + rain/steam effects
    FULL = "full"           # Complex loops, camera drift

class SceneAnimationSettings(bpy.types.PropertyGroup):
    preset: bpy.props.EnumProperty(
        items=[
            ('NONE', "None", "Static scene only"),
            ('MINIMAL', "Minimal", "Light flicker, breathing (5s loop)"),
            ('LOFI', "Lofi Stream", "Rain, light flicker, subtle motion (20s loop)"),
            ('FULL', "Full Animation", "Complex loops, camera movement"),
        ],
        default='MINIMAL'
    )

    enable_light_flicker: bpy.props.BoolProperty(default=True)
    enable_breathing: bpy.props.BoolProperty(default=True)
    enable_rain: bpy.props.BoolProperty(default=False)
    enable_camera_drift: bpy.props.BoolProperty(default=False)

    loop_duration: bpy.props.IntProperty(default=120, min=30, max=600)  # frames
```

### Animation Builders

```python
def add_light_flicker(light_obj, intensity_base, variance=0.1, speed=1.0):
    """
    Adds noise-driven flicker to light strength.
    Uses driver with noise modifier.
    """
    light = light_obj.data

    # Add driver to strength
    driver = light.driver_add("energy")
    var = driver.driver.variables.new()
    var.name = "noise"
    var.type = 'SINGLE_PROP'

    # Noise modifier
    fmod = driver.modifiers.new('NOISE')
    fmod.scale = speed
    fmod.strength = variance
    fmod.offset = 0

    driver.driver.expression = f"{intensity_base} + noise"

def add_breathing_motion(obj, scale_variance=0.002):
    """
    Subtle scale animation on Z axis (breathing effect).
    """
    obj.animation_data_create()
    action = bpy.data.actions.new(f"{obj.name}_Breathing")
    obj.animation_data.action = action

    fcurve = action.fcurves.new("scale", index=2)  # Z axis

    # Keyframes for sine wave
    for frame in range(0, 120, 10):
        value = 1.0 + scale_variance * math.sin(frame / 120 * 2 * math.pi)
        fcurve.keyframe_points.insert(frame, value)

    # Smooth interpolation
    for kf in fcurve.keyframe_points:
        kf.interpolation = 'BEZIER'

    # Loop modifier
    fmod = fcurve.modifiers.new('CYCLES')

def add_window_rain(window_obj, speed=0.5):
    """
    Animated texture on window plane (scrolling rain drops).
    """
    mat = window_obj.data.materials[0]
    nodes = mat.node_tree.nodes

    # Add noise texture for rain
    noise_tex = nodes.new('ShaderNodeTexNoise')
    noise_tex.inputs["Scale"].default_value = 50

    # Animate UV coordinates (scroll downward)
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs["Location"].driver_add("z")
    driver = mapping.inputs["Location"].driver_add("z").driver
    driver.expression = f"frame * {speed}"

    # Connect to emission
    mat.node_tree.links.new(noise_tex.outputs[0], emission_node.inputs["Strength"])
```

---

## 9. Updated Phase 1 Implementation Plan

### New Phase 1 Goals
**Duration:** 3-4 weeks
**Priority:** Visual quality over geometric accuracy

### Milestone 1.1: Worker Service Foundation (Week 1)
**Deliverables:**
- FastAPI worker running locally
- Health check endpoint
- Model loading with progress feedback:
  - Depth Anything V2
  - SAM2
  - Qwen2-VL-7B (4-bit)
- `/api/analyze` endpoint (basic version)

**Success Criteria:**
- Send image, receive depth map + masks
- Models fit in 16GB VRAM comfortably

---

### Milestone 1.2: VLM Scene Graph with Materials & Lighting (Week 1-2)
**Deliverables:**
- VLM prompt engineering for:
  - Camera estimation
  - Room planes
  - Object detection
  - **Material extraction** (albedo, roughness from image regions)
  - **Lighting hypothesis** (key direction, temperature, intensity)
- JSON schema validation
- Clarification request system (optional questions in response)

**Success Criteria:**
- VLM outputs valid scene graph JSON with material + lighting data
- Materials include sampled colors from image
- Lighting includes 3-point setup (key, fill, ambient/window)

---

### Milestone 1.3: Blender Scene Builder + Materials (Week 2-3)
**Deliverables:**
- Blender add-on UI panel
- Scene builder that creates:
  - Camera
  - Room planes
  - Proxy objects (boxes for now)
  - **Materials with extracted textures**
  - **Lights (Area lights for key/fill, Emission for window)**
- Material system:
  ```python
  def create_material_from_spec(spec):
      mat = bpy.data.materials.new(spec["target_id"])
      mat.use_nodes = True
      nodes = mat.node_tree.nodes

      bsdf = nodes["Principled BSDF"]
      bsdf.inputs["Base Color"].default_value = spec["base_color"] + [1.0]
      bsdf.inputs["Roughness"].default_value = spec["roughness"]

      # Optional: Add texture from sampled region
      if "texture_region" in spec:
          tex_node = nodes.new('ShaderNodeTexImage')
          tex_node.image = extract_texture_from_region(image, spec["texture_region"])
          mat.node_tree.links.new(tex_node.outputs[0], bsdf.inputs["Base Color"])

      return mat
  ```

**Success Criteria:**
- Scene builds successfully from JSON
- Objects have materials with correct colors
- Lights are positioned and configured
- Viewport render looks reasonably good

---

### Milestone 1.4: Eevee Render Preset (Week 3)
**Deliverables:**
- Compositing node setup for lofi aesthetic:
  ```python
  def setup_lofi_compositor():
      scene = bpy.context.scene
      scene.use_nodes = True
      tree = scene.node_tree
      tree.nodes.clear()

      # Nodes
      render_layers = tree.nodes.new('CompositorNodeRLayers')

      # Bloom
      glare = tree.nodes.new('CompositorNodeGlare')
      glare.glare_type = 'FOG_GLOW'
      glare.quality = 'HIGH'
      glare.threshold = 0.8
      glare.size = 6

      # Color correction (warm it up slightly)
      color_balance = tree.nodes.new('CompositorNodeColorBalance')
      color_balance.correction_method = 'LIFT_GAMMA_GAIN'
      color_balance.gain = (1.05, 1.0, 0.95)  # Slight warm tint

      # Vignette
      lens_distortion = tree.nodes.new('CompositorNodeLensdist')
      lens_distortion.inputs["Dispersion"].default_value = 0.02

      # Film grain
      grain = tree.nodes.new('CompositorNodeMixRGB')
      grain.blend_type = 'OVERLAY'
      grain.inputs["Fac"].default_value = 0.15

      # Composite
      composite = tree.nodes.new('CompositorNodeComposite')

      # Links
      tree.links.new(render_layers.outputs[0], glare.inputs[0])
      tree.links.new(glare.outputs[0], color_balance.inputs[1])
      tree.links.new(color_balance.outputs[0], lens_distortion.inputs[0])
      tree.links.new(lens_distortion.outputs[0], grain.inputs[1])
      tree.links.new(grain.outputs[0], composite.inputs[0])
  ```

- Eevee settings preset:
  ```python
  def setup_eevee_lofi():
      scene = bpy.context.scene
      eevee = scene.eevee

      # Render quality
      eevee.taa_render_samples = 128
      eevee.use_gtao = True
      eevee.use_bloom = True
      eevee.bloom_intensity = 0.3
      eevee.bloom_threshold = 0.8

      # Soft shadows
      eevee.use_soft_shadows = True
      eevee.shadow_cube_size = '2048'

      # Ambient occlusion
      eevee.gtao_distance = 0.5
      eevee.gtao_factor = 1.0

      # Color management
      scene.view_settings.look = 'Medium Contrast'
  ```

**Success Criteria:**
- One-button "Lofi Render Setup"
- Test render looks stylistically correct (soft, warm, cozy)
- Bloom, grain, vignette all working

---

### Milestone 1.5: Manual Refinement UI (Week 4)
**Deliverables:**
- N-panel with sliders for:
  - Camera FOV
  - Light intensities
  - Material roughness
  - Global color temperature
- Viewport gizmos for:
  - Wall plane adjustment (arrows)
  - Object repositioning (standard transform)
- "Re-analyze" button (re-run worker with current scene state)

**Success Criteria:**
- User can adjust lighting and see live preview
- Gizmos work intuitively
- Changes persist when saving .blend file

---

### Phase 1 Final Deliverable
**Test:** Load lofi girl reference image
**Result:** Blender scene that:
- Has correct camera perspective (approximate)
- Room geometry is reasonable
- Objects are placed logically
- **Materials look good** (colors match image)
- **Lighting creates the right mood** (warm key, cool fill)
- **Render output looks lofi** (bloom, grain, vignette)
- User can tweak with sliders and gizmos

**Not Required in Phase 1:**
- Perfect geometric accuracy (that's Phase 2 verification loop)
- Real furniture assets (boxes are fine)
- Animation (that's configurable later)

---

## 10. Testing Strategy Updates

### Visual Regression Testing
Since we're prioritizing visual quality, add visual tests:

```python
# tests/test_visual_quality.py
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

def test_lofi_render_quality():
    """
    Render test scene, compare to reference using SSIM.
    """
    # Build scene from test image
    scene_graph = load_test_scene_graph("lofi_girl")
    build_scene(scene_graph)

    # Render
    rendered = render_scene()

    # Load reference
    reference = Image.open("tests/references/lofi_girl_reference.png")

    # Compare (Structural Similarity Index)
    ssim_score = calculate_ssim(rendered, reference)

    # Should be > 0.7 for "looks similar"
    assert ssim_score > 0.7, f"Visual quality too low: {ssim_score}"

def test_material_color_accuracy():
    """
    Extracted material colors should match image regions.
    """
    image = load_test_image("bedroom")
    scene_graph = analyze_image(image)

    for mat_spec in scene_graph["materials"]:
        # Sample original image region
        bbox = mat_spec["sample_region"]
        sampled_color = sample_image_region(image, bbox)

        # Compare to VLM extracted color
        vlm_color = mat_spec["base_color"]

        # Colors should be close (allow some variance)
        color_diff = np.linalg.norm(np.array(sampled_color) - np.array(vlm_color))
        assert color_diff < 0.15, f"Material color too different: {color_diff}"
```

---

## 11. Configuration File (Complete User Preferences)

```yaml
# config/user_preferences.yaml
worker:
  mode: auto  # local | cloud | auto
  local_url: http://127.0.0.1:8000
  cloud_api_key: ${CLOUD_API_KEY}  # Environment variable

  vlm_backend: qwen2vl  # qwen2vl | qwen25vl | gpt4o | claude_sonnet

  gpu:
    target_vram_gb: 16
    enable_quantization: true
    quantization_bits: 4

workflow:
  automation_level: semi_automatic  # fully_automatic | guided | semi_automatic

  uncertainty_handling: questions  # auto | questions | markup | measurements

  adjustment_mode: automatic  # automatic | preview | visual_diff | manual_only

scene:
  style_preset: lofi  # lofi | realistic | stylized | architectural

  mvp_priority: visual_quality  # accuracy | speed | visual_quality

  animation:
    preset: minimal  # none | minimal | lofi | full
    enable_light_flicker: true
    enable_breathing: true
    enable_rain: false
    loop_duration_frames: 120

assets:
  library_path: ./assets

  format_priority:
    - blend
    - gltf
    - procedural

  auto_replace_proxies: true  # Replace boxes with real assets when available

rendering:
  engine: EEVEE  # EEVEE | CYCLES

  samples: 128

  compositing:
    enable_bloom: true
    enable_grain: true
    enable_vignette: true
    grain_strength: 0.15
    bloom_intensity: 0.3

verification:
  enable_in_mvp: false  # Disabled in Phase 1
  max_iterations: 5
  convergence_threshold: 0.85

development:
  cache_worker_results: true
  cache_ttl_hours: 24
  verbose_logging: true
  save_intermediate_steps: true  # Save depth maps, masks, etc for debugging
```

---

## Summary of Key Refinements

| Aspect | Original Plan | Refined Plan |
|--------|---------------|--------------|
| **Deployment** | Local-only | Hybrid (local + cloud fallback) |
| **User Control** | Fully automatic | Semi-automatic + refinement UI |
| **Uncertainty** | Best-guess only | Multi-method clarification (questions, markup, measurements) |
| **MVP Priority** | Geometric accuracy | Visual quality (materials, lighting, render) |
| **VLM Model** | Qwen2-VL only | User-selectable (local + cloud options) |
| **Assets** | .blend files only | Multi-format (blend + GLTF + procedural) |
| **Verification** | Phase 1 | Moved to Phase 2 (configurable) |
| **Animation** | Phase 6 | Configurable from Phase 1 (minimal preset) |
| **Adjustments** | Auto-apply | User-configurable behavior |
| **VRAM Target** | 8GB (aggressive) | 16GB (comfortable) |

---

## Next Implementation Steps

1. **Validate refinements with user** ✓ (you're reading this)
2. **Set up project structure** (directories, git repo)
3. **Begin Milestone 1.1** (worker service + model loading)
4. **Parallel: Build Blender add-on UI scaffold**
5. **Test integration** (HTTP communication working)
6. **Iterate on VLM prompts** (material + lighting focus)

---

**Ready to build. Choose your starting point:**
- A: Scaffold the directory structure and initialize both projects
- B: Start with worker service (get models loading first)
- C: Start with Blender add-on (UI and HTTP client first)
- D: Write VLM prompt templates (design the "brain" logic)
