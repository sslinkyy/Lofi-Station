# Robust Implementation Plan: Image-to-Scene Tool
## Resource-Optimized, Failure-Aware, Production-Ready

---

## Part 1: Resource Audit - What We're NOT Building From Scratch

### Existing Tools We MUST Leverage

#### 1. Depth Estimation (Multiple Options - Use Best)
| Model | Quality | Speed | VRAM | Notes |
|-------|---------|-------|------|-------|
| **Depth Anything V2** | Excellent | Fast | 4GB | Current choice, good generalization |
| **Marigold** | Best (diffusion) | Slow | 8GB | For high-quality mode, better detail |
| **ZoeDepth** | Good | Very Fast | 2GB | Fallback for low VRAM |
| **MiDaS v3.1** | Good | Fast | 3GB | Proven, widely used |

**Strategy:** Use Depth Anything V2 as default, Marigold for "high quality" mode, ZoeDepth as fallback

#### 2. Segmentation (Beyond Just SAM2)
| Tool | Purpose | When to Use |
|------|---------|-------------|
| **SAM2** | Instance segmentation | Primary, user can refine masks |
| **OneFormer** | Panoptic segmentation | Room parsing (floor/walls/ceiling) |
| **GroundingDINO** | Text-prompted detection | Find specific objects ("desk lamp") |
| **ODISE** | Open-vocab segmentation | Handles unusual objects |

**Strategy:** SAM2 for objects, OneFormer for room structure, GroundingDINO when VLM needs specific object

#### 3. Geometric Vision (DON'T reimplement computer vision)
| Library | Use Case | Example |
|---------|----------|---------|
| **OpenCV** | Vanishing point detection, line detection, plane fitting | Camera calibration |
| **pycolmap** | Multi-view geometry, camera pose estimation | Multi-image mode |
| **Open3D** | 3D geometry processing, mesh reconstruction | Depth to mesh conversion |
| **Trimesh** | Mesh operations, repair, simplification | Cleanup imported meshes |

**Strategy:** Use OpenCV for single-image geometry, pycolmap for multi-view

#### 4. 3D Reconstruction (Existing Pipelines)
| Tool | Type | Integration |
|------|------|-------------|
| **Gaussian Splatting** (gsplat, nerfstudio) | Scene reconstruction | Multi-view mode |
| **InstantNGP** | Fast NeRF | Alternative to splats |
| **COLMAP** | Structure from Motion | Camera pose estimation |
| **OpenMVS** | Multi-view stereo | Dense reconstruction |

**Strategy:** Use nerfstudio (includes splats + NeRF + COLMAP integration) for multi-view mode

#### 5. Material Estimation (New Research)
| Tool | Output | Use |
|------|--------|-----|
| **Intrinsic Image Decomposition** (IID) | Albedo + Shading | Separate lighting from material |
| **ControlNet + StableDiffusion** | Material maps | Generate normal/roughness maps |
| **MaterialGAN** | PBR textures | Upscale/enhance extracted textures |

**Strategy:** Use IID to get albedo, then ControlNet to generate normal/roughness if needed

#### 6. Lighting Estimation (Critical for Quality)
| Tool | Output | Accuracy |
|------|--------|----------|
| **HDRIHAVEN + matching** | Environment map | Good if scene type matches |
| **Neural Inverse Rendering** | Light positions + intensities | Best for indoor scenes |
| **Lalonde et al. (sun position)** | Outdoor lighting | Outdoor scenes only |

**Strategy:** Create lighting estimator pipeline using neural inverse rendering

#### 7. Asset Libraries (Don't Build Assets)
| Source | Format | API | Cost |
|--------|--------|-----|------|
| **Poly Haven** | GLTF, Blend | Free API | Free |
| **Sketchfab** | GLTF | API (paid) | $$ |
| **Blender Kit** | Blend | Plugin exists | Free tier |
| **Objaverse** | GLTF, OBJ | Direct download | Free |

**Strategy:** Integrate Poly Haven API, use Objaverse for fallback, optional Sketchfab

#### 8. VLM Options (Expanded)
| Model | Strengths | Cost | Our Use |
|-------|-----------|------|---------|
| **GPT-4o** | Best overall, reliable JSON | $$$ | Cloud option, verification |
| **Claude Sonnet 4.5** | Excellent vision, spatial reasoning | $$ | Cloud option, detailed analysis |
| **Qwen2-VL-7B** | Good quality, local | Free | Default local option |
| **LLaVA-NeXT-34B** | Best local quality | Free | High-quality local mode |
| **Molmo-7B** | Spatial understanding | Free | Geometric reasoning tasks |
| **CogVLM2** | Fine detail detection | Free | Material/texture analysis |

**Strategy:** Multi-VLM ensemble - different models for different tasks

#### 9. Blender Existing Add-ons (Integrate, Don't Compete)
| Add-on | Function | Integration |
|--------|----------|-------------|
| **NeRFStudio Importer** | Import splats/NeRFs | Use for multi-view mode |
| **BlenderProc** | Synthetic data generation | Generate training data |
| **AI Render** | AI upscaling/enhancement | Post-processing renders |
| **Photographer** | Camera presets | Use their camera system |

**Strategy:** Build on top of existing add-ons, not in competition

---

## Part 2: Critical Shortfalls & Mitigations

### Shortfall 1: Single Image Depth is Fundamentally Ambiguous

**Problem:**
- A 2D image → infinite possible 3D scenes
- Depth networks guess based on priors (indoor scenes, typical furniture)
- Will fail on unusual perspectives, scales, objects

**Mitigations:**
1. **Always require ONE user measurement** (mandatory in UI)
   - User clicks two points: "this bed is 6 feet wide"
   - Everything else scales from this anchor
   - Without this, scale will be wrong 80% of the time

2. **Confidence maps from depth model**
   - Depth Anything V2 can output uncertainty
   - Mark low-confidence regions in Blender (different material color)
   - VLM prioritizes high-confidence regions

3. **Geometric constraints**
   - Rooms have typical dimensions (2.4-3m ceiling height)
   - Furniture has typical sizes (bed ~2m, chair ~0.5m)
   - Use constraints as priors, flag violations

4. **Multi-view upgrade path**
   - Make it EASY to add 2-3 more images
   - Even 2 images drastically improves accuracy
   - UI button: "Add Another Angle (Recommended)"

**Implementation:**
```python
class ScaleCalibration:
    """Mandatory scale calibration system."""

    def __init__(self):
        self.anchor_measurement = None  # Required
        self.confidence_threshold = 0.7

    def require_user_measurement(self, image):
        """Block scene generation until user provides scale."""
        modal = TwoPointMeasurementModal(image)
        modal.message = "Click two points to set scale (REQUIRED for accuracy)"
        modal.show()

        # Wait for user input
        self.anchor_measurement = modal.get_result()

        if not self.anchor_measurement:
            raise ValueError("Cannot proceed without scale measurement")

    def validate_scene_scale(self, scene_graph):
        """Check if generated scales make physical sense."""
        issues = []

        for obj in scene_graph["objects"]:
            if obj["type"] == "bed" and obj["scale"][0] > 3.0:
                issues.append(f"Bed too large: {obj['scale'][0]}m (expected ~2m)")

            if obj["type"] == "door" and obj["scale"][2] < 1.8:
                issues.append(f"Door too short: {obj['scale'][2]}m (expected ~2m)")

        return issues
```

---

### Shortfall 2: VLM Hallucination & JSON Errors

**Problem:**
- VLMs make stuff up
- JSON output can be invalid, missing fields, nonsensical values
- Camera FOV might be 500°, object positions might be [NaN, NaN, NaN]

**Mitigations:**
1. **Strict JSON schema validation** (Pydantic models)
   ```python
   from pydantic import BaseModel, Field, validator

   class CameraParams(BaseModel):
       fov_deg: float = Field(ge=10, le=120)  # Enforce valid range
       pitch_deg: float = Field(ge=-90, le=90)
       yaw_deg: float = Field(ge=-180, le=180)
       position_hint: list[float] = Field(min_items=3, max_items=3)

       @validator('position_hint')
       def check_finite(cls, v):
           if not all(np.isfinite(x) for x in v):
               raise ValueError("Camera position must be finite")
           return v
   ```

2. **Multi-stage VLM prompting** (reduce complexity)
   - Stage 1: VLM analyzes image → text description
   - Stage 2: VLM converts description → JSON (simpler task)
   - Stage 3: VLM reviews its own JSON for errors

3. **Ensemble validation** (use two VLMs)
   - Qwen2-VL generates scene graph
   - GPT-4o reviews and corrects it
   - Discrepancies flagged for user review

4. **Constrained generation**
   - Use grammar-based generation (llama.cpp, outlines library)
   - Force VLM to output valid JSON schema
   - Impossible to generate invalid JSON

**Implementation:**
```python
from outlines import models, generate

# Force VLM to follow exact JSON schema
model = models.transformers("Qwen/Qwen2-VL-7B-Instruct")
schema = {
    "type": "object",
    "properties": {
        "camera": {"type": "object", "properties": {"fov_deg": {"type": "number", "minimum": 10, "maximum": 120}}},
        # ... full schema
    },
    "required": ["camera", "room", "objects"]
}

generator = generate.json(model, schema)
result = generator(prompt)  # Guaranteed valid JSON
```

---

### Shortfall 3: Occlusion (Can't See Behind Things)

**Problem:**
- Can't see back wall if bed is against it
- Can't see floor under blanket
- Missing geometry creates holes

**Mitigations:**
1. **Inpainting with depth-aware diffusion**
   - Use Stable Diffusion inpainting + depth conditioning
   - Fill occluded regions with plausible content
   - "Complete this room as if the bed wasn't there"

2. **Procedural completion**
   - Floors/walls extend automatically (simple plane fitting)
   - Occluded regions marked as "procedural guess"
   - User can adjust if wrong

3. **Multi-view solves this** (encourage users)
   - 2 photos from different angles → see behind objects
   - UI prominently suggests: "Add another photo to see occluded areas"

**Implementation:**
```python
def complete_occluded_regions(image, depth, masks):
    """Fill in parts of scene hidden by foreground objects."""

    # Identify occlusion boundaries
    occlusion_mask = detect_occlusion_boundaries(depth, masks)

    # Inpaint using depth-conditioned diffusion
    completed_image = depth_aware_inpaint(
        image=image,
        mask=occlusion_mask,
        depth_hint=depth,
        prompt="complete the room, maintain style and perspective"
    )

    # Mark these regions as "low confidence"
    return completed_image, occlusion_mask
```

---

### Shortfall 4: Lighting Estimation is Highly Underconstrained

**Problem:**
- Many different light setups can produce same image
- Single image → can't separate albedo from lighting perfectly
- Shadows might be baked into textures

**Mitigations:**
1. **Multi-VLM lighting analysis**
   - VLM 1 (Claude Sonnet): Describe lighting in natural language
     - "Warm desk lamp from left, cool window light from behind, soft ambient"
   - VLM 2 (GPT-4o): Convert description to Blender parameters
   - Structured output: light positions, temperatures, intensities

2. **Intrinsic image decomposition**
   - Separate albedo (material color) from shading (lighting effects)
   - Extract actual material colors without baked shadows
   - Tools: MIT Intrinsic Images, IIW dataset models

3. **HDRI environment matching**
   - Extract dominant light direction from shadows
   - Search HDRI database for similar lighting conditions
   - Use as starting point, let user refine

4. **Inverse rendering (if multi-view available)**
   - With 3+ photos, can estimate light positions accurately
   - Use differentiable renderer (Mitsuba 3, nvdiffrec)

**Implementation:**
```python
async def estimate_lighting_multimodal(image, depth):
    """Use multiple methods for robust lighting estimation."""

    # Method 1: VLM describes lighting
    lighting_description = await vlm_analyze_lighting(image)
    # "Warm key light from left side, approximately 45° angle, 2700K color temperature"

    # Method 2: Extract from intrinsic decomposition
    albedo, shading = decompose_intrinsic(image)
    light_dirs = analyze_shading_gradients(shading, depth)

    # Method 3: Shadow analysis
    shadow_info = detect_and_analyze_shadows(image, depth)

    # Fuse all three
    final_lighting = fuse_lighting_estimates([
        lighting_description,
        light_dirs,
        shadow_info
    ])

    return final_lighting
```

---

### Shortfall 5: Performance - Models Are Slow

**Problem:**
- Depth model: 2-5 seconds
- Segmentation: 3-8 seconds
- VLM: 10-30 seconds
- Total: 15-43 seconds (too slow)

**Mitigations:**
1. **Parallel execution** (run simultaneously, not sequentially)
   ```python
   async def analyze_image_parallel(image):
       # Run all three in parallel
       depth_task = asyncio.create_task(run_depth_model(image))
       seg_task = asyncio.create_task(run_segmentation(image))
       vlm_task = asyncio.create_task(run_vlm_analysis(image))

       # Wait for all to complete
       depth, masks, description = await asyncio.gather(
           depth_task, seg_task, vlm_task
       )

       # Now run VLM scene graph (needs results from above)
       scene_graph = await generate_scene_graph(image, depth, masks, description)

       return scene_graph
   ```

2. **Aggressive caching**
   - Hash input image → cache all results
   - Changing Blender settings doesn't re-run models
   - Only re-analyze if image changes

3. **Progressive refinement**
   - First pass: Fast models (ZoeDepth, Qwen2-VL-7B)
   - Build scene immediately (5-10 seconds)
   - Background: Run high-quality models
   - Offer "upgrade to HQ" button when ready

4. **Model quantization & optimization**
   - 4-bit quantization (GPTQ, AWQ)
   - Flash Attention 2
   - Torch compile
   - Expected speedup: 2-3x

**Implementation:**
```python
class ProgressiveAnalysisPipeline:
    """Fast preview, then high-quality refinement."""

    async def analyze_fast(self, image):
        """Quick pass with fast models."""
        start = time.time()

        depth = await self.zoedepth.predict(image)  # 1s
        masks = await self.sam2.quick_segment(image, grid_points=16)  # 2s
        scene = await self.qwen2vl.generate_scene_graph(image, depth, masks)  # 8s

        print(f"Fast preview ready in {time.time() - start:.1f}s")
        return scene, quality="preview"

    async def analyze_hq(self, image):
        """High-quality pass in background."""
        depth = await self.marigold.predict(image)  # 15s
        masks = await self.sam2.auto_segment(image, quality="high")  # 8s
        scene = await self.claude_sonnet.generate_scene_graph(image, depth, masks)  # 20s

        return scene, quality="high"
```

---

### Shortfall 6: Material Complexity (Flat Colors Aren't Enough)

**Problem:**
- Extracting median color from region → flat, boring materials
- Real materials have texture, normal detail, roughness variation
- Lofi aesthetic still needs SOME texture

**Mitigations:**
1. **Texture extraction with seamless tiling**
   - Extract texture patch from image region
   - Make tileable using Poisson blending or diffusion
   - Apply to proxy geometry

2. **Procedural augmentation**
   - Start with extracted color
   - Add procedural noise (fabric weave, wood grain)
   - Use Blender shader nodes (Noise, Voronoi, Wave)

3. **AI texture generation** (controlled)
   - Use ControlNet with extracted color as guide
   - Generate subtle normal map
   - Generate roughness variation

4. **Material library matching**
   - Extract color + rough texture type (fabric, wood, metal)
   - Search procedural material library (Blender's built-in + Poly Haven)
   - Apply matched material, tint to extracted color

**Implementation:**
```python
def create_material_from_region(image, region_bbox, material_type):
    """Create rich material, not just flat color."""

    # 1. Extract color (median)
    base_color = sample_region_color(image, region_bbox)

    # 2. Extract texture patch
    texture_patch = extract_texture(image, region_bbox)
    texture_tileable = make_seamless(texture_patch)

    # 3. Detect material type from patch
    if material_type == "fabric":
        # Generate subtle fabric normal map
        normal_map = generate_fabric_normal(texture_patch, scale=0.02)
        roughness = 0.8

    elif material_type == "wood":
        # Use procedural wood, tinted to extracted color
        shader = create_procedural_wood(base_color)
        return shader

    # 4. Build Blender material
    mat = bpy.data.materials.new(name=f"Mat_{region_bbox}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base_color + [1.0]
    bsdf.inputs["Roughness"].default_value = roughness

    # Add texture
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = bpy.data.images.new("Texture", width=512, height=512)
    tex_node.image.pixels = texture_tileable.flatten()

    # Add normal map if exists
    if normal_map is not None:
        normal_tex = nodes.new('ShaderNodeTexImage')
        normal_tex.image = load_image(normal_map)

        normal_map_node = nodes.new('ShaderNodeNormalMap')
        mat.node_tree.links.new(normal_tex.outputs[0], normal_map_node.inputs[1])
        mat.node_tree.links.new(normal_map_node.outputs[0], bsdf.inputs["Normal"])

    return mat
```

---

### Shortfall 7: Error Accumulation (Pipeline Fragility)

**Problem:**
- Bad depth → bad object placement → bad verification → worse corrections
- Errors compound through pipeline
- One wrong decision breaks everything downstream

**Mitigations:**
1. **Confidence tracking through pipeline**
   ```python
   class ConfidenceTrackedResult:
       value: Any
       confidence: float  # 0-1
       method: str  # How was this derived?
       dependencies: list  # What results does this depend on?

   # Example:
   bed_position = ConfidenceTrackedResult(
       value=[1.2, 2.0, 0.4],
       confidence=0.75,
       method="depth_map + VLM",
       dependencies=[depth_map, vlm_objects]
   )
   ```

2. **Checkpoint validation** (stop errors from propagating)
   - After depth: Check for NaN, unrealistic values
   - After segmentation: Ensure masks don't overlap incorrectly
   - After VLM: Validate JSON schema, physical constraints
   - After scene build: Check for collisions, objects below floor

3. **Staged user review** (human in the loop at checkpoints)
   - Show user depth map: "Does this look right?"
   - Show user segmentation: "Are these objects labeled correctly?"
   - Build scene: User can adjust before materials/lighting

4. **Automatic anomaly detection**
   - Flag camera FOV > 90° (likely wrong)
   - Flag objects with scale < 0.1m or > 10m (likely wrong)
   - Flag materials with roughness < 0.1 on fabric (likely wrong)

**Implementation:**
```python
class PipelineValidator:
    """Validate at each stage, prevent error propagation."""

    def validate_depth(self, depth_map):
        issues = []

        if np.any(np.isnan(depth_map)):
            issues.append("Depth map contains NaN values")

        if depth_map.max() / depth_map.min() > 100:
            issues.append("Unrealistic depth range (100:1), likely error")

        return issues

    def validate_scene_graph(self, scene_graph):
        issues = []

        # Check camera
        if scene_graph["camera"]["fov_deg"] > 90:
            issues.append(f"Camera FOV very wide: {scene_graph['camera']['fov_deg']}°")

        # Check object sizes
        for obj in scene_graph["objects"]:
            volume = np.prod(obj["world_scale"])
            if volume < 0.001:  # 1 liter
                issues.append(f"Object {obj['id']} extremely small: {volume}m³")

        # Check physical plausibility
        for obj in scene_graph["objects"]:
            if obj["world_position"][2] < -0.1:  # Below floor
                issues.append(f"Object {obj['id']} below floor level")

        return issues

    def block_if_critical(self, issues):
        critical = [i for i in issues if "NaN" in i or "below floor" in i]

        if critical:
            raise PipelineError(f"Critical validation failures: {critical}")
```

---

## Part 3: Tools We Need to CREATE

### Tool 1: Geometric Constraint Solver

**Purpose:** Ensure physically plausible scenes (no floating objects, objects inside walls, etc.)

**Features:**
- Snap objects to floor
- Prevent interpenetration
- Align to walls automatically
- Respect gravity

**Implementation:**
```python
class GeometricConstraintSolver:
    """Ensure scene is physically plausible."""

    def __init__(self, room_planes, objects):
        self.room = room_planes
        self.objects = objects

    def solve(self):
        """Apply all constraints."""
        self.snap_to_floor()
        self.resolve_collisions()
        self.align_to_walls()
        self.validate_physics()

    def snap_to_floor(self):
        """Move objects to sit on floor."""
        floor_z = self.room["floor"]["distance"]

        for obj in self.objects:
            if obj["type"] in ["bed", "desk", "chair"]:  # Furniture sits on floor
                obj_bottom = obj["position"][2] - obj["scale"][2] / 2
                offset = floor_z - obj_bottom
                obj["position"][2] += offset

    def resolve_collisions(self):
        """Move objects apart if they overlap."""
        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i+1:]:
                if self.check_collision(obj1, obj2):
                    self.separate_objects(obj1, obj2)

    def align_to_walls(self):
        """Snap objects near walls to be flush."""
        for obj in self.objects:
            if obj["type"] in ["bed", "desk"]:  # Often against walls
                nearest_wall = self.find_nearest_wall(obj)

                if self.distance_to_wall(obj, nearest_wall) < 0.3:  # Within 30cm
                    self.align_to_wall(obj, nearest_wall)
```

---

### Tool 2: Asset Matcher (Intelligent Library Search)

**Purpose:** Given VLM description, find best matching asset from library

**Features:**
- Semantic search (not just keyword)
- Style matching (lofi vs realistic)
- Size compatibility
- Fallback to procedural if no match

**Implementation:**
```python
class SemanticAssetMatcher:
    """Match VLM object descriptions to asset library."""

    def __init__(self, asset_library_path):
        self.library = self.load_library(asset_library_path)

        # Use CLIP for semantic matching
        self.clip_model = load_clip_model()

        # Precompute embeddings for all assets
        self.asset_embeddings = self.compute_asset_embeddings()

    def find_best_match(self, vlm_object_desc, style_preset):
        """
        vlm_object_desc: "modern minimalist bed with wooden frame"
        style_preset: "lofi"
        """

        # 1. Compute embedding for description
        query_embedding = self.clip_model.encode_text(vlm_object_desc)

        # 2. Find candidates by type
        object_type = vlm_object_desc.split()[0]  # "bed"
        candidates = [a for a in self.library if a["type"] == object_type]

        # 3. Score by semantic similarity
        scores = []
        for asset in candidates:
            # CLIP similarity
            semantic_score = cosine_similarity(
                query_embedding,
                self.asset_embeddings[asset["id"]]
            )

            # Style match
            style_score = asset["style_match"].get(style_preset, 0.5)

            # Combined score
            total_score = 0.7 * semantic_score + 0.3 * style_score
            scores.append((asset, total_score))

        # 4. Return best match
        if scores:
            best_asset = max(scores, key=lambda x: x[1])
            return best_asset[0]
        else:
            # Fallback to procedural
            return self.get_procedural_fallback(object_type)
```

---

### Tool 3: Training Data Generator (Synthetic Scenes)

**Purpose:** Generate synthetic Blender scenes to fine-tune models

**Why:** We can create perfect training data (known depth, segmentation, lighting) to improve our models

**Features:**
- Randomize room layouts
- Random furniture placement
- Random materials, lighting
- Export: RGB image, depth, segmentation masks, scene graph JSON

**Implementation:**
```python
class SyntheticSceneGenerator:
    """Generate training data for fine-tuning VLM."""

    def generate_scene(self):
        """Create random room scene in Blender."""

        # 1. Random room dimensions
        room_width = random.uniform(3.0, 6.0)
        room_depth = random.uniform(3.0, 5.0)
        room_height = random.uniform(2.4, 3.0)

        self.create_room(room_width, room_depth, room_height)

        # 2. Random furniture
        num_furniture = random.randint(3, 8)
        for _ in range(num_furniture):
            furniture_type = random.choice(["bed", "desk", "chair", "lamp", "plant"])
            asset = self.library.get_random_asset(furniture_type)
            position = self.find_valid_position(asset)
            self.place_asset(asset, position)

        # 3. Random materials
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                mat = self.create_random_material()
                obj.data.materials.append(mat)

        # 4. Random lighting
        self.setup_random_lighting()

        # 5. Random camera
        self.place_random_camera()

        return self.export_training_data()

    def export_training_data(self):
        """Render and export all training labels."""

        # RGB render
        rgb_image = self.render_rgb()

        # Depth map (from Blender, perfect ground truth)
        depth_map = self.render_depth()

        # Segmentation (from Blender object IDs)
        seg_masks = self.render_segmentation_masks()

        # Scene graph (we built it, so we know it exactly)
        scene_graph = self.export_scene_graph()

        return {
            "image": rgb_image,
            "depth": depth_map,
            "masks": seg_masks,
            "scene_graph": scene_graph  # Perfect label for VLM training
        }
```

**Use:** Generate 10,000 synthetic scenes → fine-tune Qwen2-VL → better scene graph accuracy

---

### Tool 4: Confidence Scorer (Know When to Ask for Help)

**Purpose:** Automatically detect when tool is uncertain, prompt user proactively

**Features:**
- Analyzes all pipeline outputs
- Computes confidence scores
- Decides when to ask user vs. make best guess

**Implementation:**
```python
class ConfidenceScorer:
    """Determine when to ask user for clarification."""

    def score_depth_confidence(self, depth_map, variance_map):
        """Use model's built-in uncertainty if available."""
        if variance_map is not None:
            # High variance = low confidence
            mean_variance = variance_map.mean()
            confidence = 1.0 - min(mean_variance / 0.5, 1.0)
        else:
            # Heuristic: smooth gradients = high confidence
            gradients = np.gradient(depth_map)
            smoothness = 1.0 / (np.std(gradients) + 1e-6)
            confidence = min(smoothness / 10.0, 1.0)

        return confidence

    def score_object_confidence(self, vlm_object, depth_map, mask):
        """Compute confidence in detected object."""

        # 1. VLM's stated confidence
        vlm_conf = vlm_object.get("confidence", 0.5)

        # 2. Mask quality (crisp edges = high confidence)
        mask_crispness = self.compute_mask_crispness(mask)

        # 3. Depth consistency (uniform depth inside mask = high confidence)
        depth_variance = depth_map[mask].std()
        depth_conf = 1.0 - min(depth_variance / 2.0, 1.0)

        # Combined
        total_conf = (vlm_conf * 0.5 + mask_crispness * 0.25 + depth_conf * 0.25)

        return total_conf

    def should_ask_user(self, confidence, importance):
        """Decide whether to interrupt workflow."""

        # High importance + low confidence = ask
        if importance > 0.7 and confidence < 0.6:
            return True

        # Low importance + very low confidence = ask
        if confidence < 0.3:
            return True

        # Otherwise proceed with best guess
        return False
```

---

### Tool 5: Benchmark Suite (Automated Testing)

**Purpose:** Test on diverse images automatically, track performance over time

**Features:**
- Test image database (indoor, outdoor, different styles)
- Ground truth where available (synthetic scenes)
- Metrics: depth accuracy, segmentation IoU, scene graph correctness
- Regression detection

**Implementation:**
```python
class BenchmarkSuite:
    """Automated testing on diverse images."""

    def __init__(self):
        self.test_cases = [
            {"image": "lofi_girl.png", "type": "stylized_indoor", "has_ground_truth": False},
            {"image": "bedroom_01.png", "type": "realistic_indoor", "has_ground_truth": False},
            {"image": "synthetic_room_001.png", "type": "synthetic", "has_ground_truth": True},
            # ... 100+ test cases
        ]

    async def run_full_benchmark(self):
        """Run entire pipeline on all test cases."""

        results = []
        for test_case in self.test_cases:
            print(f"Testing: {test_case['image']}")

            # Run pipeline
            start = time.time()
            scene_graph = await self.pipeline.analyze(test_case["image"])
            duration = time.time() - start

            # Evaluate if ground truth available
            if test_case["has_ground_truth"]:
                metrics = self.evaluate_against_ground_truth(
                    scene_graph,
                    test_case["ground_truth"]
                )
            else:
                # Manual inspection required
                metrics = {"requires_manual_review": True}

            results.append({
                "test_case": test_case["image"],
                "duration": duration,
                "metrics": metrics
            })

        self.generate_report(results)
        return results

    def evaluate_against_ground_truth(self, predicted, ground_truth):
        """Compare predicted scene graph to known correct answer."""

        metrics = {}

        # Camera accuracy
        fov_error = abs(predicted["camera"]["fov_deg"] - ground_truth["camera"]["fov_deg"])
        metrics["camera_fov_error"] = fov_error

        # Object detection (precision/recall)
        pred_objects = set(o["type"] for o in predicted["objects"])
        true_objects = set(o["type"] for o in ground_truth["objects"])

        metrics["object_precision"] = len(pred_objects & true_objects) / len(pred_objects)
        metrics["object_recall"] = len(pred_objects & true_objects) / len(true_objects)

        # Position accuracy (mean error)
        position_errors = []
        for pred_obj in predicted["objects"]:
            # Find matching ground truth object
            true_obj = next((o for o in ground_truth["objects"] if o["id"] == pred_obj["id"]), None)
            if true_obj:
                pos_error = np.linalg.norm(
                    np.array(pred_obj["world_position"]) - np.array(true_obj["world_position"])
                )
                position_errors.append(pos_error)

        metrics["mean_position_error"] = np.mean(position_errors) if position_errors else None

        return metrics
```

---

## Part 4: Revised Architecture (Resource-Optimized)

### Parallel Multi-Model Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                    IMAGE INPUT                            │
└───────────────────┬──────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┬──────────────────┐
    │               │               │                  │
    ▼               ▼               ▼                  ▼
┌────────┐    ┌──────────┐   ┌──────────┐      ┌──────────┐
│ Depth  │    │   SAM2   │   │OneFormer │      │  VLM-1   │
│Anything│    │(objects) │   │  (room)  │      │(describe)│
└───┬────┘    └────┬─────┘   └────┬─────┘      └────┬─────┘
    │              │              │                  │
    └──────────────┼──────────────┼──────────────────┘
                   │              │
                   ▼              ▼
            ┌──────────────────────────┐
            │   Confidence Scorer      │
            └──────────┬───────────────┘
                       │
                ┌──────┴───────┐
                │              │
                ▼              ▼
         ┌──────────┐   ┌────────────┐
         │ VLM-2    │   │  User      │
         │(scene    │◄──┤Clarification│
         │ graph)   │   │ (if needed)│
         └────┬─────┘   └────────────┘
              │
              ▼
       ┌──────────────┐
       │  Validator   │
       │  (Pydantic)  │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Constraint  │
       │   Solver     │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Scene Builder│
       │  + Asset     │
       │   Matcher    │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Material    │
       │  Generator   │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Lighting    │
       │  Estimator   │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Blender     │
       │   Scene      │
       └──────────────┘
```

**Key improvements:**
1. Parallel execution at top (depth + segmentation + VLM description)
2. Confidence scorer gates user clarification
3. Validator prevents bad data from propagating
4. Constraint solver ensures physical plausibility
5. Asset matcher uses semantic search
6. Material/lighting are first-class components

---

## Part 5: Actionable Implementation Roadmap

### Phase 0: Infrastructure (Week 1)

**Goal:** Set up all foundations before writing pipeline code

**Tasks:**
1. ✅ Set up git repo with proper structure
2. ✅ Create AI worker service skeleton (FastAPI)
3. ✅ Set up Blender add-on template
4. ✅ Download and test all models locally:
   - Depth Anything V2
   - SAM2
   - OneFormer
   - Qwen2-VL-7B
5. ✅ Set up async pipeline orchestration (asyncio)
6. ✅ Create Pydantic models for all JSON schemas
7. ✅ Build confidence scorer framework
8. ✅ Build validator framework
9. ✅ Set up caching system (Redis or file-based)
10. ✅ Create benchmark test harness

**Deliverable:** Empty pipeline that can pass images through, with all infrastructure working

---

### Phase 1: Core Perception (Week 2)

**Goal:** Get raw data from models, validated and confidence-scored

**Tasks:**
1. Implement parallel depth + segmentation + description
2. Integrate Depth Anything V2 with confidence maps
3. Integrate SAM2 + OneFormer fusion
4. Build first VLM prompt (image description, not scene graph yet)
5. Implement confidence scorer for all outputs
6. Build validator with automatic anomaly detection
7. Test on 20 diverse images, collect failure modes

**Deliverable:** Given image, get back validated depth, masks, description with confidence scores

---

### Phase 2: Scene Graph Generation (Week 3)

**Goal:** VLM converts perception data to structured scene graph

**Tasks:**
1. Write VLM prompt for scene graph (with materials + lighting)
2. Implement constrained JSON generation (outlines library)
3. Build ensemble validator (two VLMs cross-check)
4. Implement geometric constraint solver
5. Build user clarification system (questions + visual markup)
6. Test scene graph generation on 50 images
7. Fine-tune on synthetic data if accuracy < 80%

**Deliverable:** Validated, physically-plausible scene graph JSON

---

### Phase 3: Asset System (Week 4)

**Goal:** Convert scene graph objects to actual 3D assets

**Tasks:**
1. Build asset library catalog system
2. Integrate Poly Haven API
3. Implement semantic asset matcher (CLIP-based)
4. Create procedural fallback generators (bed, chair, desk, lamp)
5. Build asset importer (blend + GLTF)
6. Implement asset placement with constraint solving

**Deliverable:** Scene graph → Blender scene with real assets (not just boxes)

---

### Phase 4: Materials & Lighting (Week 5)

**Goal:** High-quality materials and accurate lighting

**Tasks:**
1. Implement intrinsic image decomposition
2. Build texture extractor + seamless tiling
3. Create procedural material library (fabric, wood, metal)
4. Build multi-method lighting estimator
5. Implement 3-point lighting setup
6. Create material/lighting refinement UI (sliders in Blender)

**Deliverable:** Scenes that render with professional quality

---

### Phase 5: Blender Integration & UX (Week 6)

**Goal:** Polished add-on with all features accessible

**Tasks:**
1. Build main UI panel (clean, intuitive)
2. Implement scale calibration UI (two-point measurement)
3. Build visual markup mode for clarifications
4. Create viewport gizmos for room adjustment
5. Implement progressive refinement (fast preview → HQ upgrade)
6. Add "Re-analyze" and "Verify" buttons
7. Build export system (.blend + textures + report)

**Deliverable:** Full working add-on, user-testable

---

### Phase 6: Verification & Polish (Week 7-8)

**Goal:** Iterative refinement, edge case handling

**Tasks:**
1. Implement verification loop (render → compare → adjust)
2. Build visual diff system
3. Add multi-view mode (basic NeRF integration)
4. Handle edge cases (outdoor scenes, unusual objects)
5. Performance optimization (caching, quantization, parallel processing)
6. User testing with 10+ beta testers
7. Fix bugs, refine UX based on feedback

**Deliverable:** Production-ready tool

---

## Part 6: Success Metrics (How We Know It Works)

### Quantitative Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Depth Accuracy** | RMSE < 0.3m on indoor scenes | Benchmark on synthetic data with known depth |
| **Object Detection** | Recall > 85%, Precision > 90% | Manual labeling of 100 test images |
| **Camera FOV Error** | < 10° on average | Compare to manual measurement |
| **Scene Graph Valid Rate** | > 95% pass validation | Automated schema validation |
| **Position Accuracy** | Mean error < 0.4m | Synthetic scenes with ground truth |
| **Pipeline Speed** | < 20s total (fast mode) | Benchmark on standard hardware |
| **User Satisfaction** | 4+/5 average rating | Beta tester survey |

### Qualitative Metrics

- Renders "look right" to untrained eye (80%+ agreement in blind test)
- Materials match image aesthetic
- Lighting creates correct mood
- Scenes are editable without breaking

---

## Part 7: Risk Mitigation Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| VLM produces invalid JSON | High | High | Constrained generation + Pydantic validation |
| Depth model fails on unusual images | Medium | High | Multi-model ensemble + user fallback |
| Scale is completely wrong | High | Medium | Mandatory user measurement |
| Performance too slow for users | Medium | Medium | Progressive refinement + caching |
| Assets don't match style | Low | Medium | Semantic search + procedural fallback |
| Lighting looks wrong | Medium | High | Multi-method estimation + user adjustment sliders |
| Users can't fix errors | Low | High | Full manual refinement UI with gizmos |
| Tool doesn't work on user's hardware | Low | Critical | Cloud fallback mode |

---

## Summary: What Makes This Plan Robust

### ✅ Resource Optimization
- Uses existing models (not training from scratch)
- Leverages libraries (OpenCV, Open3D, not reinventing)
- Integrates existing Blender add-ons (not competing)
- Accesses free asset libraries (Poly Haven, Objaverse)

### ✅ Failure Awareness
- Confidence scoring throughout pipeline
- Validation at every stage (prevent error propagation)
- User clarification when uncertain
- Geometric constraints ensure physical plausibility
- Multiple fallback strategies (ensemble models, procedural generation)

### ✅ Tool Creation
- Constraint solver (ensures plausible scenes)
- Asset matcher (semantic search, not keyword)
- Training data generator (fine-tune on synthetic scenes)
- Confidence scorer (know when to ask for help)
- Benchmark suite (catch regressions)

### ✅ Actionable Roadmap
- 8-week plan with weekly deliverables
- Each phase builds on previous
- Parallel work streams (worker + add-on)
- Early testing (don't wait until end)

### ✅ Success Metrics
- Quantitative benchmarks (depth RMSE, object recall)
- Qualitative assessment (renders look good)
- User satisfaction (beta testing)

---

## Next Step: Choose Starting Point

**Option A: Full Infrastructure First** (Recommended)
- Set up repos, models, frameworks
- Build empty pipeline with all validators/scorers
- THEN fill in actual perception/generation
- Pro: Solid foundation, easier to debug
- Con: Takes longer to see results

**Option B: Vertical Slice** (Fastest to demo)
- Build minimal end-to-end: image → depth → simple scene
- Add validation/confidence/assets later
- Pro: Working demo in days
- Con: More refactoring later

**Option C: Critical Path** (Balanced)
- Week 1: Infrastructure + depth/seg models working
- Week 2: VLM scene graph with validation
- Week 3: Blender scene builder with basic assets
- Week 4+: Materials, lighting, UX, polish
- Pro: Balanced risk, regular progress
- Con: Requires discipline to not skip validation

**Which approach do you want?**
