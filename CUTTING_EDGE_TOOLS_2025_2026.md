# Cutting-Edge AI Tools for Image-to-Scene (2025-2026)
## Research-Backed Tool Recommendations

Based on latest research from trusted AI sources (January 2026), here are the state-of-the-art tools we should use:

---

## 1. Monocular Depth Estimation

### **Apple Depth Pro** (2024) 🔥 NEW RECOMMENDATION
- **Performance:** Sharp monocular metric depth in **< 1 second**
- **Key Feature:** Precise focal length estimation (critical for camera matching)
- **Status:** Open-source (Apple Machine Learning Research)
- **Use:** Primary depth model (replaces Depth Anything V2 as default)

**Source:** [Apple ML Research - Depth Pro](https://machinelearning.apple.com/research/depth-pro)

### **Metric3D v2** (2024) 🔥 NEW
- **Type:** Versatile monocular geometric foundation model
- **Output:** Zero-shot metric depth + surface normals
- **Use:** High-quality mode, provides surface normals for better reconstruction

### **Depth Any Camera (DAC)** (2025) 🔥 EMERGING
- **Key Feature:** Works with fisheye and 360° cameras (non-perspective)
- **Use:** Edge case handler for unusual camera types
- **Innovation:** Extends perspective models without task-specific training

**Source:** [Papers with Code - Monocular Depth Estimation](https://paperswithcode.com/task/monocular-depth-estimation)

### **Marigold** (CVPR 2024)
- **Type:** Diffusion-based depth estimation
- **Quality:** Best detail, but slower
- **Use:** "Ultra HQ" mode for final production

**Updated Stack:**
```python
DEPTH_MODELS = {
    "fast": "apple/depth-pro",           # < 1s, NEW DEFAULT
    "balanced": "metric3d-v2",            # + surface normals
    "hq": "marigold",                     # Diffusion, slow but best
    "edge_case": "depth-any-camera"       # Fisheye/360° support
}
```

---

## 2. Vision Language Models (VLMs)

### **Qwen3-VL** (2025) 🔥 BRAND NEW - TOP TIER
- **Model Size:** Qwen3-VL-235B-A22B-Instruct (flagship)
- **Performance:** **Rivals GPT-5 and Gemini 2.5 Pro** on multimodal benchmarks
- **Features:**
  - Stronger multimodal reasoning
  - Agentic capabilities
  - Long-context comprehension
  - Covers: Q&A, 2D/3D grounding, video, OCR, documents
- **Use:** **PRIMARY LOCAL VLM** (upgrade from Qwen2-VL)

**Source:** [DataCamp - Top VLMs 2026](https://www.datacamp.com/blog/top-vision-language-models)

### **GLM-4.6V** (2025) 🔥 NEW
- **Developer:** Z.ai (GLM family)
- **Features:**
  - Native multimodal tool use
  - Stronger visual reasoning
  - 128K context window
- **Use:** Alternative local option, tool use integration

**Source:** [Hugging Face - VLMs 2025](https://huggingface.co/blog/vlms-2025)

### **Gemini 2.5 Pro** (2025) 🔥 UPDATED
- **Context:** **> 1 million tokens** (2M coming soon)
- **Strengths:** Outstanding reasoning across complex image scenes
- **Use:** Cloud option for complex scene analysis
- **Applications:** Technical inspections, product catalogs, robot vision

### **GPT-4.1** (2025) 🔥 NEW
- **Family:** GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano
- **Improvements:** Outperforms GPT-4o across benchmarks
- **Vision:** Enhanced chart/diagram analysis, visual math, object counting
- **Use:** Cloud option, good JSON compliance

### **DeepSeek-OCR** (2025) 🔥 SPECIALIZED
- **Innovation:** Contexts Optical Compression (20× compression, 97% accuracy)
- **Performance:** Outperforms GOT-OCR2.0 and MinerU2.0
- **Use:** Specialized model for text/document extraction from scenes

### **FastVLM** (Apple, 2025) 🔥 NEW
- **Speed:** Real-time applications, on-device capable
- **Architecture:** Hybrid visual encoder for high-res images
- **Performance:** 361 FPS for some tasks
- **Use:** Real-time preview mode, mobile deployment

**Source:** [Apple ML Research - FastVLM](https://machinelearning.apple.com/research/fast-vision-language-models)

### **Reasoning VLMs** (2025) 🔥 EMERGING
- **QVQ-72B-preview** (Qwen) - Multimodal reasoning
- **Kimi-VL-A3B-Thinking** (Moonshot AI) - Another reasoning option
- **Use:** Complex spatial reasoning tasks

**Updated VLM Stack:**
```python
VLM_MODELS = {
    "local_flagship": "qwen3-vl-235b",          # NEW: Rivals GPT-5
    "local_balanced": "glm-4.6v",               # NEW: 128K context
    "local_fast": "apple/fastvlm",              # NEW: Real-time
    "cloud_best": "gemini-2.5-pro",             # 1M+ context
    "cloud_fast": "gpt-4.1",                    # Improved GPT-4o
    "specialized_ocr": "deepseek-ocr",          # Text extraction
    "reasoning": "qvq-72b-preview"              # Complex spatial reasoning
}
```

---

## 3. Image-to-3D Scene Reconstruction

### **Apple SHARP** (December 2025) 🔥 GAME CHANGER
- **Speed:** **< 1 second** to reconstruct 3D scene from single image
- **Method:** Regresses 3D Gaussian representation parameters
- **Hardware:** Single feedforward pass on standard GPU
- **Status:** **Open-source on GitHub**
- **Use:** **CORE TECHNOLOGY** - this is exactly what we need

**Impact:** This could replace our entire depth → mesh pipeline with native 3D Gaussian output

**Source:** [Apple SHARP Research](https://9to5mac.com/2025/12/17/apple-sharp-ai-model-turns-2d-photos-into-3d-views/)

### **Cornell C3Po** (December 2025) 🔥 NEW
- **Purpose:** Links photos to floor plans with pixel-level accuracy
- **Dataset:** C3 - 90,000 floor-plan/photo pairs, 153M pixel correspondences
- **Performance:** 34% error reduction vs. best previous method
- **Use:** Room layout extraction, floor plan generation

**Source:** [Cornell Tech News - C3Po](https://tech.cornell.edu/news/c3po/)

### **Flash3D** (June 2025, updated)
- **Efficiency:** Trainable on single GPU in one day
- **Performance:** State-of-the-art on RealEstate10k
- **Use:** Fast training for custom datasets

**Source:** [arXiv - Flash3D](https://arxiv.org/abs/2406.04343)

### **Meta SAM 3D** (November 2025) 🔥 NEW
- **Released:** November 19, 2025
- **Capability:** Reconstruct objects AND human bodies from flat images
- **Output:** Geometry, texture, and pose
- **Use:** Character/object extraction from scenes

**Source:** [Meta SAM 3D Guide](https://www.adwaitx.com/meta-sam-3d-models-guide/)

**Updated Reconstruction Stack:**
```python
RECONSTRUCTION_METHODS = {
    "single_image_fast": "apple/sharp",              # < 1s, 3D Gaussians
    "single_image_quality": "flash3d",               # State-of-the-art
    "floor_plan": "cornell/c3po",                    # Room layout
    "objects_humans": "meta/sam-3d",                 # Object extraction
    "multi_view": "nerfstudio"                       # Multiple images
}
```

---

## 4. Image Segmentation

### **SAM2 Variants** (2024-2025) 🔥 EVOLVED ECOSYSTEM

**SAM2LONG** (2025)
- **Purpose:** Addresses error accumulation in long videos
- **Innovation:** Tree-based memory with constrained prompting
- **Use:** Video input mode

**SAMURAI** (2025)
- **Innovation:** Motion-aware memory selection
- **Performance:** Improved tracking without additional training
- **Use:** Dynamic scene segmentation

**SAMWISE** (2025)
- **Features:** Natural language understanding + temporal modeling
- **Use:** Text-prompted segmentation ("segment the bed")

**Language Segment-Anything**
- **Feature:** Language prompts instead of bounding boxes
- **Use:** VLM-integrated segmentation pipeline

**Source:** [Exploring SAM2 Variants](https://www.sievedata.com/resources/exploring-sam2-variants)

### **FastSAM** (2025)
- **Speed:** **50× faster than original SAM**
- **FPS:** > 30 frames per second
- **Use:** Real-time applications, preview mode

### **OneFormer** (2024)
- **Capability:** Semantic + Instance + Panoptic in ONE model
- **Training:** Single training, multiple tasks
- **Use:** Room structure parsing (floor/walls/ceiling/furniture)

**Source:** [Best Segmentation Models 2025](https://averroes.ai/blog/best-image-segmentation-models)

**Updated Segmentation Stack:**
```python
SEGMENTATION_MODELS = {
    "objects": "sam2",                        # Default for objects
    "objects_fast": "fastsam",                # Real-time (50x faster)
    "room_structure": "oneformer",            # Semantic segmentation
    "language_guided": "samwise",             # Text prompts
    "video": "sam2long"                       # Long videos
}
```

---

## 5. Gaussian Splatting (2025 Improvements)

### **Depth Prior Integration** (2025) 🔥 MAJOR IMPROVEMENT
- **Method:** Leverage Depth-Anything V2 as depth prior
- **Performance:** **38% faster training** with depth supervision
- **Quality:** Better geometric accuracy
- **Use:** Integrate depth maps into splat optimization

**Source:** [Enhanced 3DGS with Depth Priors](https://www.mdpi.com/1424-8220/25/22/6999)

### **MILo** (Mesh-in-the-Loop) (2025) 🔥 NEW
- **Innovation:** Differentiable mesh extraction DURING optimization
- **Benefit:** Gradient flow from mesh to Gaussians
- **Result:** Bidirectional consistency between volume and surface
- **Use:** Export clean meshes directly from Gaussians

**Source:** [MILo Project Page](https://anttwo.github.io/milo/)

### **Geometric Enhancement** (2025)
- **Addresses:** Floating artifacts, incomplete surfaces in sparse-view
- **Methods:**
  - Side-view Inconsistency Filtering (SIF)
  - Local Depth Regularization (LDR)
  - Anisotropy-aware Shape Regularization (ASR)
- **Use:** Better geometry from limited views

### **Human Reconstruction** (2025)
- **Speed:** Real-time rendering up to 361 FPS
- **Methods:** SiTH, PSHuman, PARTE
- **Integration:** Diffusion models for occluded view inference
- **Use:** Character extraction and reconstruction

**Updated Gaussian Splatting Stack:**
```python
GAUSSIAN_SPLAT_CONFIG = {
    "depth_prior": "depth-anything-v2",           # 38% faster training
    "mesh_extraction": "milo",                    # Differentiable mesh
    "sparse_view_enhancement": True,              # Geometric regularization
    "human_reconstruction": "psihuman",           # Character handling
}
```

---

## 6. Material & Lighting Estimation

### **IDArb** (ICLR 2025 submission) 🔥 CUTTING EDGE
- **Method:** Diffusion-based intrinsic decomposition
- **Input:** Arbitrary number of images with varying illuminations
- **Output:** Multi-view consistent surface normals + material properties
- **Use:** Multi-view material estimation (when available)

**Source:** [IDArb - OpenReview](https://openreview.net/forum?id=uuef1HP6X7)

### **RGB↔X** (SIGGRAPH 2024) 🔥 BIDIRECTIONAL
- **Direction 1 (RGB → X):** Estimates albedo, roughness, metallicity, lighting
- **Direction 2 (X → RGB):** Synthesizes realistic images from intrinsic channels
- **Use:** Material estimation + validation (re-synthesize and compare)

**Source:** [RGB↔X - SIGGRAPH 2024](https://dl.acm.org/doi/10.1145/3641519.3657445)

### **Intrinsic Image Fusion** (December 2024)
- **Method:** Multi-view physically-based material reconstruction
- **Feature:** Diffusion-based material estimator
- **Output:** Multiple candidate decompositions per view
- **Use:** Ensemble material estimation

**Source:** [Intrinsic Image Fusion - arXiv](https://arxiv.org/html/2512.13157v1)

### **Colorful Diffuse Intrinsic Decomposition** (TOG 2024)
- **Repository:** Available on GitHub (compphoto/Intrinsic)
- **Feature:** "Colorful" - handles colored illumination better
- **Use:** Scenes with complex lighting (warm + cool lights)

**Source:** [Intrinsic Decomposition - GitHub](https://github.com/compphoto/Intrinsic)

**Updated Material/Lighting Stack:**
```python
MATERIAL_LIGHTING_MODELS = {
    "single_view": "rgb-x",                       # SIGGRAPH 2024
    "multi_view": "idarb",                        # ICLR 2025, multi-view consistent
    "fusion": "intrinsic-image-fusion",           # Ensemble estimation
    "colored_light": "colorful-diffuse-iid"       # Complex lighting
}
```

---

## 7. Existing Image-to-Blender Pipelines

### **Commercial Tools with Blender Integration**

**Meshy AI** (2025)
- **Integration:** Native Blender, Unity, Unreal, Maya plugins
- **Performance:** GPU-accelerated on NVIDIA RTX + Apple Silicon
- **Export:** GLB, OBJ
- **Use:** Asset generation, not full scene reconstruction

**Tripo AI** (2025)
- **Specialty:** Game-optimized quad-based topology
- **Speed:** Fast generation
- **Use:** Individual asset creation

**Source:** [10 AI 3D Tools for 2026](https://www.3daistudio.com/3d-generator-ai-comparison-alternatives-guide/best-3d-generation-tools-2026/10-ai-3d-tools-supercharge-blender-unreal-workflow-2026)

### **Blender Add-ons**

**Pic To 3D Mesh** (Updated 2025)
- **Depth:** MiDaS v3.1 pipeline (improved edge fidelity)
- **Features:** Auto-retopology, watertight mesh export
- **Use:** Quick depth-to-mesh conversion

### **Nvidia AI Blueprint** (2025)
- **Direction:** REVERSE - Blender scenes → AI image generation
- **Purpose:** Structure AI-generated images using Blender layouts
- **Insight:** Shows Blender integration is production-ready

**Source:** [Nvidia AI Blueprint](https://www.creativebloq.com/3d/new-nvidia-ai-blueprint-is-a-bridge-between-blender-and-ai-image-generation)

---

## 8. 3D Scene Understanding & Geometric Reasoning

### **3D Scene Question Answering (3D SQA)** (Survey, Feb 2026) 🔥 EMERGING FIELD
- **Purpose:** Spatial understanding + multimodal reasoning
- **Use:** VLM validation ("Is the bed against the wall?")
- **Applications:** Embodied intelligence, robotic perception

**Source:** [3D SQA Survey - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006967)

### **Talk2PC** (2025)
- **Method:** Fuses LiDAR + radar through prompt-guided cross-attention
- **Use:** 3D grounding in driving scenes
- **Insight:** Cross-modal fusion for better accuracy

### **AI City Challenge 2025**
- **Focus:** Warehouse-scale 3D scene understanding via natural language
- **Goal:** Integrate visual perception + geometric reasoning + language
- **Relevance:** Benchmark for our scene understanding quality

**Source:** [AI City Challenge 2025](https://www.aicitychallenge.org/2025-track3/)

---

## 9. COMPLETELY NEW APPROACHES WE SHOULD CONSIDER

### **Approach 1: Direct 3D Gaussian Output (Apple SHARP)**

**Instead of:**
```
Image → Depth Map → Mesh → Materials → Blender
```

**Use:**
```
Image → SHARP (< 1s) → 3D Gaussians → Blender (native support)
```

**Advantages:**
- 10× faster
- Native 3D representation
- Blender already has Gaussian import add-ons
- Can convert Gaussians to mesh later if needed (MILo)

**Implementation:**
```python
# New primary pipeline
def analyze_image_sharp(image):
    # Apple SHARP: image → 3D Gaussians in < 1 second
    gaussians = sharp_model.predict(image)

    # Optional: Extract mesh using MILo
    mesh = milo_extract_mesh(gaussians)

    # VLM still provides scene graph for:
    # - Object labels
    # - Room planes
    # - Material assignments
    # - Lighting setup

    scene_graph = vlm_analyze(image, gaussians)

    return gaussians, scene_graph
```

### **Approach 2: Multi-VLM Ensemble (Specialized Experts)**

**Instead of:** One VLM does everything

**Use:** Different VLMs for different tasks
```python
EXPERT_VLMS = {
    "spatial_reasoning": "qwen3-vl-235b",           # Scene structure
    "ocr_text": "deepseek-ocr",                     # Text in scene
    "material_analysis": "rgb-x",                    # Material properties
    "lighting_description": "gemini-2.5-pro",        # Lighting analysis
    "verification": "gpt-4.1"                        # Quality check
}
```

**Advantages:**
- Each model does what it's best at
- Higher overall accuracy
- Redundancy for critical decisions

### **Approach 3: Hybrid Depth (Multi-Model Fusion)**

**Instead of:** Pick one depth model

**Use:** Ensemble of multiple models
```python
def get_robust_depth(image):
    # Run three models in parallel
    depth_sharp = depth_pro.predict(image)           # Fast, metric
    depth_metric3d = metric3d.predict(image)         # + normals
    depth_marigold = marigold.predict(image)         # Best detail

    # Fuse using confidence maps
    depth_fused = adaptive_fusion([
        (depth_sharp, sharp_confidence),
        (depth_metric3d, metric3d_confidence),
        (depth_marigold, marigold_confidence)
    ])

    return depth_fused
```

**Advantages:**
- More robust than single model
- Reduces artifacts
- Can use faster models for preview, slower for final

---

## 10. UPDATED TECHNOLOGY STACK (FINAL RECOMMENDATIONS)

### Primary Pipeline (Single Image, Fast Mode)

```python
FAST_PIPELINE = {
    "3d_reconstruction": "apple/sharp",              # < 1s → Gaussians
    "scene_understanding": "qwen3-vl-235b",          # Rivals GPT-5
    "segmentation": "oneformer",                     # Room structure
    "material": "rgb-x",                             # Intrinsic decomp
    "lighting": "gemini-2.5-pro"                     # Cloud, complex reasoning
}
```

**Timeline:** 2-5 seconds total

### High-Quality Pipeline (Best Results)

```python
HQ_PIPELINE = {
    "depth": "ensemble(depth-pro, metric3d-v2, marigold)",  # Multi-model fusion
    "3d_reconstruction": "sharp + milo",             # Gaussians → clean mesh
    "scene_understanding": "qwen3-vl-235b",          # Local VLM
    "segmentation": "sam2 + oneformer",              # Objects + structure
    "material": "idarb",                             # Multi-view if available
    "lighting": "intrinsic-fusion",                  # Multi-method
    "verification": "gpt-4.1"                        # Cloud validation
}
```

**Timeline:** 20-40 seconds

### Multi-View Pipeline (Accurate Geometry)

```python
MULTIVIEW_PIPELINE = {
    "floor_plan": "cornell/c3po",                    # Layout extraction
    "reconstruction": "nerfstudio + depth-priors",   # Gaussian splat with depth
    "mesh_extraction": "milo",                       # Differentiable mesh
    "material": "idarb",                             # Multi-view intrinsic
    "scene_graph": "qwen3-vl-235b",                  # Semantic understanding
}
```

**Timeline:** 2-5 minutes (depending on number of images)

---

## 11. CRITICAL UPDATES TO ORIGINAL PLAN

### What Changes Based on New Research?

**MAJOR CHANGE 1: Apple SHARP as Core Technology**

**Old Plan:**
```
Depth Anything V2 → Depth Map → Open3D mesh → Blender
```

**New Plan:**
```
Apple SHARP → 3D Gaussians → Blender (< 1 second)
```

**Impact:**
- 10× faster
- Better quality (native 3D, not derived from depth)
- Can still extract depth/mesh if needed

**MAJOR CHANGE 2: Qwen3-VL (not Qwen2-VL)**

**Old:** Qwen2-VL-7B
**New:** Qwen3-VL-235B (if VRAM allows) or Qwen3-VL-7B

**Reason:** Rivals GPT-5, much better multimodal reasoning

**MAJOR CHANGE 3: Multi-VLM Ensemble (not single VLM)**

**Old:** One VLM does everything
**New:** Specialized VLMs for different tasks

**Specialists:**
- Spatial reasoning: Qwen3-VL
- OCR: DeepSeek-OCR
- Material analysis: RGB↔X integration
- Verification: GPT-4.1

**MAJOR CHANGE 4: Floor Plan Extraction (Cornell C3Po)**

**New capability:** Extract floor plan from photo
- 34% better than previous methods
- Pixel-level accuracy
- Could generate 2D layout before 3D reconstruction

**Use:** Constraint for room dimensions, validate 3D reconstruction

**MAJOR CHANGE 5: Real-time Preview (FastVLM + SHARP)**

**New workflow:**
1. User uploads image
2. SHARP generates Gaussians (< 1s)
3. FastVLM quick analysis (< 1s)
4. Show preview in Blender (2s total)
5. Background: Run HQ pipeline
6. Offer "Upgrade to HQ" button when ready

**User Experience:** Near-instant feedback, then progressive enhancement

---

## 12. INTEGRATION PRIORITIES (WHAT TO BUILD FIRST)

### Phase 0.5: Validate New Technologies (NEW - Week 0)

**Before building full pipeline, test new tools:**

1. ✅ Test Apple SHARP
   - Can we run it?
   - Quality vs depth-based methods?
   - Blender import workflow?

2. ✅ Test Qwen3-VL
   - VRAM requirements?
   - Scene graph output quality?
   - Quantization strategy?

3. ✅ Test Cornell C3Po
   - Floor plan extraction quality?
   - Integration with 3D reconstruction?

4. ✅ Benchmark FastVLM
   - Speed vs quality tradeoff?
   - Good enough for preview mode?

**Deliverable:** Technology validation report, decide which tools make the cut

### Updated Phase 1: Core Pipeline with New Tech (Week 1-2)

**Build around Apple SHARP + Qwen3-VL:**

```python
# Minimum viable pipeline
def analyze_image_v2(image):
    # 1. SHARP: Fast 3D reconstruction
    gaussians = sharp_model.predict(image)

    # 2. Qwen3-VL: Scene understanding
    scene_graph = qwen3vl.generate_scene_graph(image, gaussians)

    # 3. Build Blender scene
    build_blender_scene(gaussians, scene_graph)
```

**Timeline:** 2 weeks to working prototype

---

## 13. RISK ASSESSMENT OF NEW TECHNOLOGIES

| Technology | Maturity | Risk | Mitigation |
|------------|----------|------|------------|
| Apple SHARP | Just released (Dec 2025) | High - might have bugs | Keep depth-based fallback |
| Qwen3-VL | Very new (2025) | Medium - VRAM heavy | Test quantization, fallback to Qwen2-VL |
| Cornell C3Po | Research (Dec 2025) | High - may not be released | Optional feature, not critical path |
| MILo | Recent (2025) | Medium - mesh quality unknown | Alternative: traditional mesh extraction |
| IDArb | ICLR submission (2025) | High - not published yet | Use RGB↔X instead |

**Strategy:** Build with fallbacks
- Primary: New cutting-edge tools
- Fallback: Proven 2024 tools (Depth Anything V2, Qwen2-VL, SAM2)

---

## 14. FINAL RECOMMENDATION

### Option A: Bleeding Edge (Highest Risk, Best Potential)

Use ALL new 2025-2026 tools:
- Apple SHARP
- Qwen3-VL
- Cornell C3Po
- MILo
- IDArb

**Pro:** Best possible results if everything works
**Con:** Many untested technologies, high failure risk

### Option B: Balanced Innovation (RECOMMENDED)

Mix proven + cutting-edge:
- **3D Reconstruction:** Apple SHARP (proven to work)
- **VLM:** Qwen3-VL with Qwen2-VL fallback
- **Segmentation:** SAM2 (proven) + OneFormer
- **Materials:** RGB↔X (SIGGRAPH, more mature than IDArb)
- **Depth (if needed):** Depth Pro (Apple, proven)

**Pro:** Best new tech where it matters, safety net elsewhere
**Con:** Not using absolute cutting edge everywhere

### Option C: Conservative (Lowest Risk)

Stick with 2024 proven tech, add only:
- Depth Pro (Apple, well-tested)
- Qwen2.5-VL (more mature than Qwen3)
- OneFormer (add to SAM2)

**Pro:** All proven technologies
**Con:** Missing out on major improvements (SHARP, Qwen3-VL)

---

## VERDICT: Option B (Balanced Innovation)

**Rationale:**
- Apple SHARP is game-changing, worth the risk
- Qwen3-VL has clear advantages, but keep Qwen2-VL fallback
- C3Po/IDArb are too new, skip for Phase 1
- Focus on getting core pipeline working with best-validated new tools

**Core Tech Stack (Final):**
```python
PRODUCTION_STACK = {
    # Core (NEW)
    "3d_reconstruction": "apple/sharp",
    "vlm_primary": "qwen3-vl-7b",
    "vlm_fallback": "qwen2.5-vl-7b",
    "vlm_cloud": "gemini-2.5-pro",

    # Supporting (proven)
    "depth_backup": "depth-pro",              # If SHARP fails
    "segmentation": ["sam2", "oneformer"],
    "materials": "rgb-x",
    "mesh_extraction": "milo",

    # Fast preview
    "preview_vlm": "apple/fastvlm",
    "preview_3d": "sharp"
}
```

This gives us cutting-edge performance with safety nets.

---

## Sources

- [Apple ML Research - Depth Pro](https://machinelearning.apple.com/research/depth-pro)
- [Apple ML Research - FastVLM](https://machinelearning.apple.com/research/fast-vision-language-models)
- [Apple SHARP - 3D Reconstruction](https://9to5mac.com/2025/12/17/apple-sharp-ai-model-turns-2d-photos-into-3d-views/)
- [Cornell C3Po Research](https://tech.cornell.edu/news/c3po/)
- [DataCamp - Top VLMs 2026](https://www.datacamp.com/blog/top-vision-language-models)
- [Hugging Face - VLMs 2025](https://huggingface.co/blog/vlms-2025)
- [Papers with Code - Monocular Depth](https://paperswithcode.com/task/monocular-depth-estimation)
- [Best Image Segmentation Models 2025](https://averroes.ai/blog/best-image-segmentation-models)
- [SAM2 Variants Exploration](https://www.sievedata.com/resources/exploring-sam2-variants)
- [Enhanced 3DGS with Depth Priors](https://www.mdpi.com/1424-8220/25/22/6999)
- [MILo Project Page](https://anttwo.github.io/milo/)
- [RGB↔X SIGGRAPH 2024](https://dl.acm.org/doi/10.1145/3641519.3657445)
- [IDArb - ICLR 2025](https://openreview.net/forum?id=uuef1HP6X7)
- [Intrinsic Decomposition GitHub](https://github.com/compphoto/Intrinsic)
- [3D AI Tools 2026](https://www.3daistudio.com/3d-generator-ai-comparison-alternatives-guide/best-3d-generation-tools-2026/10-ai-3d-tools-supercharge-blender-unreal-workflow-2026)
- [Meta SAM 3D](https://www.adwaitx.com/meta-sam-3d-models-guide/)
- [Flash3D arXiv](https://arxiv.org/abs/2406.04343)
- [3D SQA Survey](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006967)
- [AI City Challenge 2025](https://www.aicitychallenge.org/2025-track3/)
