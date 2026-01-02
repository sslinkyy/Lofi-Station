"""
Image-to-Scene AI Worker
FastAPI service for AI-powered 3D scene reconstruction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import uvicorn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Image-to-Scene AI Worker",
    description="AI-powered 3D scene reconstruction from images",
    version="0.1.0"
)

# CORS middleware (allow Blender to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnalyzeRequest(BaseModel):
    """Request model for image analysis"""
    mode: str = Field(default="fast", description="fast | balanced | hq")
    style_preset: str = Field(default="lofi", description="lofi | realistic | architectural")
    reference_measurement: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User-provided scale: {px_coords: [[x1,y1], [x2,y2]], real_world_meters: float}"
    )

class CameraParams(BaseModel):
    """Camera parameters"""
    fov_deg: float = Field(ge=10, le=120)
    pitch_deg: float = Field(ge=-90, le=90)
    yaw_deg: float = Field(ge=-180, le=180)
    roll_deg: float = Field(default=0, ge=-180, le=180)
    position_hint: List[float] = Field(min_length=3, max_length=3)

class RoomPlane(BaseModel):
    """Room plane definition"""
    name: str
    normal: List[float] = Field(min_length=3, max_length=3)
    distance: float

class SceneObject(BaseModel):
    """Scene object definition"""
    id: str
    type: str
    bbox_px: List[int] = Field(min_length=4, max_length=4)
    world_position: List[float] = Field(min_length=3, max_length=3)
    world_rotation: List[float] = Field(min_length=3, max_length=3)
    world_scale: List[float] = Field(min_length=3, max_length=3)
    proxy_type: str
    confidence: float = Field(ge=0, le=1)

class SceneGraph(BaseModel):
    """Complete scene graph"""
    camera: CameraParams
    room: Dict[str, Any]
    objects: List[SceneObject]
    lighting: Dict[str, Any]
    materials: List[Dict[str, Any]]

class AnalyzeResponse(BaseModel):
    """Response from image analysis"""
    status: str
    depth_map_base64: Optional[str] = None
    segmentation_masks: Optional[Dict[str, str]] = None
    scene_graph: Optional[SceneGraph] = None
    processing_time_s: float
    model_info: Dict[str, str]


# Import our models and services
from models import DepthModelFactory, VLMFactory, SegmentationFactory
from models.segmentation_model import SegmentationService
from services import DepthService


# Global state for models (loaded on startup)
class ModelState:
    """Container for loaded models"""
    def __init__(self):
        self.depth_model = None
        self.depth_service = None
        self.sam2_model = None
        self.oneformer_model = None
        self.segmentation_service = None
        self.vlm_model = None
        self.loaded = False

models = ModelState()


# Startup event - load models
@app.on_event("startup")
async def load_models():
    """Load AI models on startup"""
    logger.info("=" * 60)
    logger.info("Loading AI models...")
    logger.info("=" * 60)

    try:
        # Determine device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Load Depth Model
        logger.info("\n[1/3] Loading Depth Model...")
        try:
            models.depth_model = DepthModelFactory.create("depth-pro", device=device)
            models.depth_service = DepthService(
                depth_model=models.depth_model,
                cache_dir="cache/depth",
                enable_cache=True
            )
            logger.info("✓ Depth Pro loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load Depth Pro: {e}")
            logger.info("  Make sure Depth Pro is installed:")
            logger.info("  git clone https://github.com/apple/ml-depth-pro external/depth-pro")
            logger.info("  cd external/depth-pro && pip install -e .")
            raise

        # Load VLM
        logger.info("\n[2/3] Loading VLM...")
        try:
            models.vlm_model = VLMFactory.create(
                model_name="qwen3-vl-8b",
                device=device,
                quantization="4bit"
            )
            logger.info("✓ Qwen3-VL loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load VLM: {e}")
            logger.info("  VLM will auto-download from Hugging Face on first use")
            logger.info("  Make sure you have internet connection and ~8GB disk space")
            # Don't raise - VLM is optional for depth-only mode
            logger.warning("Continuing without VLM (depth-only mode)")

        # Load Segmentation
        logger.info("\n[3/3] Loading Segmentation...")
        try:
            # Load SAM2 for object masks
            logger.info("  Loading SAM2...")
            try:
                models.sam2_model = SegmentationFactory.create(
                    "sam2-base",  # Use base for balance of speed/quality
                    device=device
                )
                logger.info("  ✓ SAM2 loaded successfully")
            except Exception as e:
                logger.error(f"  ✗ Failed to load SAM2: {e}")
                logger.info("  Continuing without SAM2")
                models.sam2_model = None

            # Load OneFormer for room structure
            logger.info("  Loading OneFormer...")
            try:
                models.oneformer_model = SegmentationFactory.create(
                    "oneformer-semantic",
                    device=device
                )
                logger.info("  ✓ OneFormer loaded successfully")
            except Exception as e:
                logger.error(f"  ✗ Failed to load OneFormer: {e}")
                logger.info("  Continuing without OneFormer")
                models.oneformer_model = None

            # Create segmentation service
            if models.sam2_model or models.oneformer_model:
                models.segmentation_service = SegmentationService(
                    sam2_model=models.sam2_model,
                    oneformer_model=models.oneformer_model
                )
                logger.info("✓ Segmentation service initialized")
            else:
                logger.warning("⚠️  No segmentation models loaded")

        except Exception as e:
            logger.error(f"✗ Failed to initialize segmentation: {e}")
            logger.warning("Continuing without segmentation")

        models.loaded = True
        logger.info("\n" + "=" * 60)
        logger.info("✓ All models loaded successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n✗ Failed to load models: {e}")
        models.loaded = False
        # Don't raise - let server start but mark as unhealthy
        logger.warning("Server will start but /api/analyze will not work until models are loaded")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models.loaded,
        "version": "0.1.0"
    }


# Main analysis endpoint
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(...),
    settings: Optional[str] = None
):
    """
    Analyze an image and generate scene graph

    Args:
        image: Uploaded image file
        settings: JSON string of AnalyzeRequest

    Returns:
        AnalyzeResponse with depth, segmentation, and scene graph
    """
    import time
    import base64
    from io import BytesIO
    from PIL import Image

    start_time = time.time()

    try:
        # Parse settings
        if settings:
            import json
            settings_dict = json.loads(settings)
            request_settings = AnalyzeRequest(**settings_dict)
        else:
            request_settings = AnalyzeRequest()

        # Read image
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data))

        logger.info(f"Received image: {pil_image.size}, mode: {request_settings.mode}")

        # Check if models are loaded
        if not models.loaded or models.depth_service is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Check server logs for errors."
            )

        # Step 1: Depth Estimation
        logger.info("Step 1: Estimating depth...")
        depth_result = await models.depth_service.estimate_depth(
            pil_image,
            return_confidence=True,
            mode=request_settings.mode
        )

        depth_map = depth_result["depth"]
        logger.info(f"✓ Depth estimated (range: {depth_result['min_depth']:.2f} - {depth_result['max_depth']:.2f}m)")

        # Encode depth map to base64
        depth_normalized = models.depth_service.normalize_depth(depth_map, 0, 255)
        depth_image = Image.fromarray(depth_normalized.astype('uint8'))
        depth_buffer = BytesIO()
        depth_image.save(depth_buffer, format='PNG')
        depth_base64 = base64.b64encode(depth_buffer.getvalue()).decode('utf-8')

        # Step 2: Segmentation
        logger.info("Step 2: Segmentation...")
        segmentation_masks = {}
        room_structure = None

        if models.segmentation_service is not None:
            try:
                # Get room structure (walls, floor, ceiling)
                if models.oneformer_model is not None:
                    logger.info("  Extracting room structure...")
                    room_data = models.segmentation_service.segment_scene(
                        pil_image,
                        mode="room_structure"
                    )
                    room_structure = room_data.get("room_structure")
                    logger.info(f"  ✓ Found {len(room_structure) if room_structure else 0} room elements")

                # Generate automatic object masks with SAM2
                if models.sam2_model is not None:
                    logger.info("  Running SAM2 automatic segmentation...")
                    masks_data = models.segmentation_service.segment_scene(
                        pil_image,
                        mode="auto"
                    )

                    # Convert SAM2 masks to base64 for transmission
                    import base64
                    from io import BytesIO

                    sam2_masks = masks_data.get("masks", [])
                    logger.info(f"  ✓ Generated {len(sam2_masks)} object masks")

                    # Store top N masks (by area) to avoid overwhelming the response
                    max_masks = 20
                    sorted_masks = sorted(sam2_masks, key=lambda m: m.get("area", 0), reverse=True)[:max_masks]

                    for i, mask in enumerate(sorted_masks):
                        # Convert mask to image
                        mask_array = mask["segmentation"]
                        mask_image = Image.fromarray((mask_array * 255).astype('uint8'))

                        # Encode to base64
                        mask_buffer = BytesIO()
                        mask_image.save(mask_buffer, format='PNG')
                        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')

                        segmentation_masks[f"mask_{i:03d}"] = {
                            "data": mask_base64,
                            "bbox": mask.get("bbox", [0, 0, 0, 0]),
                            "area": mask.get("area", 0),
                            "predicted_iou": mask.get("predicted_iou", 0.0),
                            "stability_score": mask.get("stability_score", 0.0)
                        }

                logger.info(f"✓ Segmentation complete ({len(segmentation_masks)} masks)")

            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                segmentation_masks = {}
                room_structure = None
        else:
            logger.info("⚠️  Segmentation service not loaded, skipping")

        # Step 3: VLM Scene Graph
        logger.info("Step 3: VLM Scene Graph...")
        if models.vlm_model is not None:
            try:
                # Convert depth map to PIL Image for VLM
                depth_pil = Image.fromarray(depth_normalized.astype('uint8'))

                scene_graph = models.vlm_model.generate_scene_graph(
                    image=pil_image,
                    depth_map=depth_pil,
                    masks=segmentation_masks,
                    style_preset=request_settings.style_preset
                )

                logger.info(f"✓ Scene graph generated with {len(scene_graph.get('objects', []))} objects")

            except Exception as e:
                logger.error(f"VLM scene graph generation failed: {e}")
                scene_graph = None
        else:
            logger.info("⚠️  VLM not loaded, skipping scene graph")
            scene_graph = None

        processing_time = time.time() - start_time
        logger.info(f"✓ Processing complete in {processing_time:.2f}s")

        # Determine status based on what components completed
        if scene_graph and segmentation_masks:
            status = "complete"  # All 3 components: depth + segmentation + VLM
        elif scene_graph:
            status = "depth_vlm"  # Depth + VLM only
        elif segmentation_masks:
            status = "depth_segmentation"  # Depth + segmentation only
        else:
            status = "depth_only"  # Only depth worked

        return AnalyzeResponse(
            status=status,
            depth_map_base64=depth_base64,
            segmentation_masks=segmentation_masks,
            scene_graph=scene_graph,
            processing_time_s=processing_time,
            model_info={
                "depth": f"{models.depth_model.get_info()['name']} (loaded)" if models.depth_model else "not loaded",
                "vlm": f"{models.vlm_model.get_info()['name']} (loaded)" if models.vlm_model else "not loaded",
                "sam2": f"{models.sam2_model.get_info()['name']} (loaded)" if models.sam2_model else "not loaded",
                "oneformer": f"{models.oneformer_model.get_info()['name']} (loaded)" if models.oneformer_model else "not loaded"
            }
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Verification endpoint (for iterative refinement)
@app.post("/api/verify")
async def verify_scene(
    original_image: UploadFile = File(...),
    rendered_image: UploadFile = File(...),
    scene_graph: str = None
):
    """
    Compare original vs rendered image and suggest adjustments

    Args:
        original_image: Original reference image
        rendered_image: Blender render
        scene_graph: Current scene graph JSON

    Returns:
        Suggested adjustments
    """
    # TODO: Implement verification logic
    return {
        "status": "not_implemented",
        "match_score": 0.0,
        "adjustments": []
    }


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
