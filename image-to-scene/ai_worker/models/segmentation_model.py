"""
Segmentation model wrappers
Supports: SAM2, OneFormer, FastSAM
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SAM2Model:
    """
    Segment Anything Model 2 (SAM2) wrapper
    https://github.com/facebookresearch/segment-anything-2
    """

    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda"
    ):
        """
        Initialize SAM2 model

        Args:
            model_size: Model size (tiny/small/base/large)
            device: Device (cuda/cpu)
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self.predictor = None

    def load(self):
        """Load SAM2 model"""
        try:
            logger.info(f"Loading SAM2 ({self.model_size})...")

            # Try to import SAM2
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                logger.error("SAM2 not installed. Install with:")
                logger.error("  git clone https://github.com/facebookresearch/segment-anything-2 external/sam2")
                logger.error("  cd external/sam2 && pip install -e .")
                raise

            # Model checkpoint mapping
            model_configs = {
                "tiny": "sam2_hiera_t.yaml",
                "small": "sam2_hiera_s.yaml",
                "base": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml"
            }

            checkpoint_urls = {
                "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
                "small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
                "base": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
                "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
            }

            config = model_configs[self.model_size]
            checkpoint = f"checkpoints/sam2_{self.model_size}.pt"

            # Build model
            self.model = build_sam2(config, checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)

            logger.info(f"✓ SAM2 ({self.model_size}) loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")
            raise

    def set_image(self, image: Image.Image):
        """
        Set image for prediction

        Args:
            image: PIL Image
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to numpy array
        image_np = np.array(image.convert('RGB'))
        self.predictor.set_image(image_np)

    def predict_masks(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Dict[str, Any]:
        """
        Predict masks from prompts

        Args:
            point_coords: Nx2 array of point prompts
            point_labels: N array of labels (1=foreground, 0=background)
            box: [x1, y1, x2, y2] bounding box
            multimask_output: Return 3 masks with different quality/granularity

        Returns:
            Dictionary with masks, scores, logits
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )

        return {
            "masks": masks,  # (N, H, W) boolean array
            "scores": scores,  # (N,) confidence scores
            "logits": logits  # (N, H, W) raw logits
        }

    def automatic_mask_generation(
        self,
        image: Image.Image,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Automatic mask generation (no prompts needed)

        Args:
            image: PIL Image
            points_per_side: Grid points per side
            pred_iou_thresh: Predicted IoU threshold
            stability_score_thresh: Stability threshold

        Returns:
            List of mask dictionaries
        """
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            logger.error("SAM2 automatic mask generator not available")
            return []

        # Create automatic mask generator
        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh
        )

        # Generate masks
        image_np = np.array(image.convert('RGB'))
        masks = mask_generator.generate(image_np)

        return masks

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": f"SAM2-{self.model_size}",
            "version": "2.0",
            "device": self.device,
            "loaded": self.model is not None,
            "capabilities": ["interactive_segmentation", "automatic_segmentation", "box_prompts"]
        }


class OneFormerModel:
    """
    OneFormer for semantic/instance/panoptic segmentation
    https://github.com/SHI-Labs/OneFormer
    """

    def __init__(
        self,
        task: str = "panoptic",
        dataset: str = "ade20k",
        device: str = "cuda"
    ):
        """
        Initialize OneFormer

        Args:
            task: Task type (semantic/instance/panoptic)
            dataset: Dataset (coco/ade20k/cityscapes)
            device: Device (cuda/cpu)
        """
        self.task = task
        self.dataset = dataset
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """Load OneFormer model"""
        try:
            logger.info(f"Loading OneFormer ({self.task} on {self.dataset})...")
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

            # Model name mapping
            model_names = {
                ("semantic", "ade20k"): "shi-labs/oneformer_ade20k_swin_large",
                ("instance", "coco"): "shi-labs/oneformer_coco_swin_large",
                ("panoptic", "ade20k"): "shi-labs/oneformer_ade20k_swin_large",
                ("panoptic", "coco"): "shi-labs/oneformer_coco_swin_large"
            }

            model_name = model_names.get((self.task, self.dataset))
            if not model_name:
                model_name = "shi-labs/oneformer_ade20k_swin_large"
                logger.warning(f"Using default model: {model_name}")

            # Load processor and model
            self.processor = OneFormerProcessor.from_pretrained(model_name)
            self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ OneFormer loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load OneFormer: {e}")
            raise

    def predict(
        self,
        image: Image.Image,
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict segmentation masks

        Args:
            image: PIL Image
            task: Override task type

        Returns:
            Segmentation results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        task = task or self.task

        # Prepare inputs
        inputs = self.processor(
            images=image,
            task_inputs=[task],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]

        # Convert to numpy
        segmentation_map = results.cpu().numpy()

        return {
            "segmentation_map": segmentation_map,  # (H, W) with class IDs
            "task": task,
            "dataset": self.dataset
        }

    def get_room_structure(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract room structure (walls, floor, ceiling)

        Args:
            image: PIL Image

        Returns:
            Room structure with masks for architectural elements
        """
        result = self.predict(image, task="semantic")
        seg_map = result["segmentation_map"]

        # ADE20K class IDs for architectural elements
        class_mapping = {
            "floor": [4, 29],  # floor, rug
            "wall": [1],  # wall
            "ceiling": [6],  # ceiling
            "window": [9],  # window
            "door": [15],  # door
            "furniture": [7, 8, 11, 19, 23, 25, 27, 31]  # various furniture
        }

        room_masks = {}
        for element, class_ids in class_mapping.items():
            mask = np.isin(seg_map, class_ids)
            if mask.any():
                room_masks[element] = mask

        return {
            "room_masks": room_masks,
            "segmentation_map": seg_map,
            "dataset": self.dataset
        }

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": f"OneFormer-{self.task}",
            "version": "1.0",
            "device": self.device,
            "dataset": self.dataset,
            "loaded": self.model is not None,
            "capabilities": ["semantic", "instance", "panoptic", "room_structure"]
        }


class SegmentationFactory:
    """Factory for creating segmentation models"""

    @staticmethod
    def create(
        model_name: str = "sam2",
        device: str = "cuda",
        **kwargs
    ):
        """
        Create segmentation model by name

        Args:
            model_name: Model identifier (sam2/oneformer)
            device: Device to run on
            **kwargs: Additional model-specific arguments

        Returns:
            Segmentation model instance
        """
        models = {
            "sam2": SAM2Model,
            "sam2-tiny": lambda device: SAM2Model(model_size="tiny", device=device),
            "sam2-small": lambda device: SAM2Model(model_size="small", device=device),
            "sam2-base": lambda device: SAM2Model(model_size="base", device=device),
            "sam2-large": lambda device: SAM2Model(model_size="large", device=device),
            "oneformer": OneFormerModel,
            "oneformer-semantic": lambda device: OneFormerModel(task="semantic", dataset="ade20k", device=device),
            "oneformer-panoptic": lambda device: OneFormerModel(task="panoptic", dataset="ade20k", device=device),
        }

        if model_name not in models:
            raise ValueError(
                f"Unknown segmentation model: {model_name}. "
                f"Available: {list(models.keys())}"
            )

        model_class = models[model_name]

        # Handle lambda constructors
        if callable(model_class) and not isinstance(model_class, type):
            model = model_class(device)
        else:
            model = model_class(device=device, **kwargs)

        model.load()
        return model


class SegmentationService:
    """
    High-level segmentation service combining SAM2 and OneFormer
    """

    def __init__(
        self,
        sam2_model: Optional[SAM2Model] = None,
        oneformer_model: Optional[OneFormerModel] = None
    ):
        """
        Initialize segmentation service

        Args:
            sam2_model: SAM2 model instance
            oneformer_model: OneFormer model instance
        """
        self.sam2 = sam2_model
        self.oneformer = oneformer_model

    def segment_scene(
        self,
        image: Image.Image,
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Segment entire scene

        Args:
            image: PIL Image
            mode: Segmentation mode (auto/interactive/room_structure)

        Returns:
            Segmentation results
        """
        results = {
            "masks": [],
            "room_structure": None,
            "segmentation_map": None
        }

        if mode == "auto" and self.sam2 is not None:
            # Use SAM2 automatic mask generation
            logger.info("Running SAM2 automatic mask generation...")
            masks = self.sam2.automatic_mask_generation(image)
            results["masks"] = masks

        if mode == "room_structure" and self.oneformer is not None:
            # Extract room structure
            logger.info("Extracting room structure with OneFormer...")
            room_data = self.oneformer.get_room_structure(image)
            results["room_structure"] = room_data["room_masks"]
            results["segmentation_map"] = room_data["segmentation_map"]

        return results

    def get_object_masks(
        self,
        image: Image.Image,
        bboxes: List[List[int]]
    ) -> List[np.ndarray]:
        """
        Get object masks from bounding boxes using SAM2

        Args:
            image: PIL Image
            bboxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of binary masks
        """
        if self.sam2 is None:
            logger.warning("SAM2 not loaded, cannot generate object masks")
            return []

        self.sam2.set_image(image)

        masks = []
        for bbox in bboxes:
            result = self.sam2.predict_masks(
                box=np.array(bbox),
                multimask_output=False
            )
            # Take the first (and only) mask
            mask = result["masks"][0]
            masks.append(mask)

        return masks
