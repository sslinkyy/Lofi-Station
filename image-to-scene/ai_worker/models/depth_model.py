"""
Depth estimation model wrapper
Supports: Depth Pro (Apple), Metric3D v2, and future models
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DepthProModel:
    """
    Apple Depth Pro model wrapper
    https://github.com/apple/ml-depth-pro
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize Depth Pro model

        Args:
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        self.model = None
        self.transform = None

    def load(self):
        """Load model weights"""
        try:
            logger.info("Loading Depth Pro model...")

            # Import depth_pro (assumes it's installed)
            # git clone https://github.com/apple/ml-depth-pro
            # cd ml-depth-pro && pip install -e .
            try:
                import depth_pro
            except ImportError:
                raise ImportError(
                    "Depth Pro not installed. Install with:\n"
                    "git clone https://github.com/apple/ml-depth-pro external/depth-pro\n"
                    "cd external/depth-pro && pip install -e ."
                )

            # Load model and preprocessing transform
            self.model, self.transform = depth_pro.create_model_and_transforms()
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ Depth Pro loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Depth Pro: {e}")
            raise

    def predict(
        self,
        image: Image.Image,
        return_confidence: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict depth map from image

        Args:
            image: PIL Image (RGB)
            return_confidence: Whether to return confidence map

        Returns:
            Tuple of (depth_map, confidence_map)
            - depth_map: HxW numpy array with metric depth values
            - confidence_map: HxW numpy array (if requested)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Preprocess image
            image_tensor = self.transform(image).to(self.device)

            # Run inference
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)

            # Extract depth
            depth = prediction["depth"].cpu().numpy()

            # Resize to original image size if needed
            if depth.shape != (image.height, image.width):
                from scipy.ndimage import zoom
                scale_h = image.height / depth.shape[0]
                scale_w = image.width / depth.shape[1]
                depth = zoom(depth, (scale_h, scale_w), order=1)

            # Confidence map (if model provides it)
            confidence = None
            if return_confidence and "confidence" in prediction:
                confidence = prediction["confidence"].cpu().numpy()
                if confidence.shape != (image.height, image.width):
                    confidence = zoom(confidence, (scale_h, scale_w), order=1)

            return depth, confidence

        except Exception as e:
            logger.error(f"Depth prediction failed: {e}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "depth-pro",
            "version": "1.0",
            "device": self.device,
            "loaded": self.model is not None,
            "output": "metric_depth",
            "speed": "< 1s on GPU"
        }


class Metric3Dv2Model:
    """
    Metric3D v2 model wrapper
    https://github.com/YvanYin/Metric3D
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load(self):
        """Load Metric3D v2 model"""
        try:
            logger.info("Loading Metric3D v2 model...")

            # Load via PyTorch Hub
            self.model = torch.hub.load(
                'YvanYin/Metric3D',
                'metric3d_v2',
                pretrained=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ Metric3D v2 loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Metric3D v2: {e}")
            raise

    def predict(
        self,
        image: Image.Image,
        return_normals: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict depth (and optionally normals) from image

        Args:
            image: PIL Image (RGB)
            return_normals: Whether to return surface normals

        Returns:
            Tuple of (depth_map, normals)
            - depth_map: HxW metric depth
            - normals: HxWx3 surface normals (if requested)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Convert to tensor
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            image_tensor = transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)

            # Extract depth
            depth = outputs['depth'].squeeze().cpu().numpy()

            # Extract normals if requested
            normals = None
            if return_normals and 'normals' in outputs:
                normals = outputs['normals'].squeeze().cpu().numpy()
                # Transpose from CxHxW to HxWxC
                normals = normals.transpose(1, 2, 0)

            return depth, normals

        except Exception as e:
            logger.error(f"Metric3D prediction failed: {e}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "metric3d-v2",
            "version": "2.0",
            "device": self.device,
            "loaded": self.model is not None,
            "output": "metric_depth + normals",
            "speed": "~2-3s on GPU"
        }


class DepthModelFactory:
    """Factory for creating depth models"""

    @staticmethod
    def create(model_name: str, device: str = "cuda"):
        """
        Create depth model by name

        Args:
            model_name: Model name (depth-pro | metric3d-v2)
            device: Device to run on

        Returns:
            Depth model instance
        """
        models = {
            "depth-pro": DepthProModel,
            "metric3d-v2": Metric3Dv2Model,
        }

        if model_name not in models:
            raise ValueError(
                f"Unknown depth model: {model_name}. "
                f"Available: {list(models.keys())}"
            )

        model = models[model_name](device=device)
        model.load()
        return model
