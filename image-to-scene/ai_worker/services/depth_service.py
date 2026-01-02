"""
Depth estimation service
Handles depth map generation with caching and error handling
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import hashlib
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DepthService:
    """Service for depth estimation"""

    def __init__(
        self,
        depth_model,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True
    ):
        """
        Initialize depth service

        Args:
            depth_model: Loaded depth model instance
            cache_dir: Directory for caching results
            enable_cache: Whether to cache results
        """
        self.model = depth_model
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/depth")
        self.enable_cache = enable_cache

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, image: Image.Image) -> str:
        """Generate cache key from image"""
        # Hash image data
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # Include model name
        model_name = self.model.get_info()["name"]

        return f"{model_name}_{img_hash}"

    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Load cached depth map"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"✓ Loaded depth from cache: {cache_key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        return None

    def _save_to_cache(
        self,
        cache_key: str,
        depth: np.ndarray,
        confidence: Optional[np.ndarray]
    ):
        """Save depth map to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((depth, confidence), f)
            logger.info(f"✓ Saved depth to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    async def estimate_depth(
        self,
        image: Image.Image,
        return_confidence: bool = False,
        mode: str = "fast"
    ) -> Dict[str, Any]:
        """
        Estimate depth from image

        Args:
            image: PIL Image (RGB)
            return_confidence: Whether to return confidence map
            mode: Processing mode (fast/balanced/hq)

        Returns:
            Dict containing:
                - depth: HxW numpy array
                - confidence: HxW numpy array (optional)
                - metadata: processing info
        """
        try:
            # Check cache
            if self.enable_cache:
                cache_key = self._get_cache_key(image)
                cached = self._load_from_cache(cache_key)

                if cached is not None:
                    depth, confidence = cached
                    return {
                        "depth": depth,
                        "confidence": confidence,
                        "cached": True,
                        "model": self.model.get_info()["name"]
                    }

            # Predict depth
            logger.info(f"Estimating depth (mode: {mode})...")
            depth, confidence = self.model.predict(
                image,
                return_confidence=return_confidence
            )

            # Validate depth
            if np.any(np.isnan(depth)):
                logger.warning("Depth map contains NaN values, replacing with median")
                depth = np.nan_to_num(depth, nan=np.median(depth[~np.isnan(depth)]))

            # Save to cache
            if self.enable_cache:
                self._save_to_cache(cache_key, depth, confidence)

            return {
                "depth": depth,
                "confidence": confidence,
                "cached": False,
                "model": self.model.get_info()["name"],
                "min_depth": float(depth.min()),
                "max_depth": float(depth.max()),
                "mean_depth": float(depth.mean())
            }

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise

    def get_depth_statistics(self, depth: np.ndarray) -> Dict[str, float]:
        """Compute depth map statistics"""
        return {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
            "median": float(np.median(depth)),
            "std": float(depth.std()),
        }

    def normalize_depth(
        self,
        depth: np.ndarray,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> np.ndarray:
        """Normalize depth map to specified range"""
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max - depth_min < 1e-6:
            logger.warning("Depth range too small, returning zeros")
            return np.zeros_like(depth)

        normalized = (depth - depth_min) / (depth_max - depth_min)
        normalized = normalized * (max_val - min_val) + min_val

        return normalized

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        camera_intrinsics: Optional[np.ndarray] = None,
        image: Optional[Image.Image] = None
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud

        Args:
            depth: HxW depth map
            camera_intrinsics: 3x3 camera matrix (optional, will estimate if None)
            image: Optional RGB image for colors

        Returns:
            Nx3 or Nx6 point cloud (XYZ or XYZRGB)
        """
        h, w = depth.shape

        # Estimate camera intrinsics if not provided
        if camera_intrinsics is None:
            # Assume typical camera parameters
            focal_length = max(h, w)
            cx, cy = w / 2, h / 2
            camera_intrinsics = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ])

        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        depth_flat = depth.flatten()

        # Back-project to 3D
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        x = (u - cx) * depth_flat / fx
        y = (v - cy) * depth_flat / fy
        z = depth_flat

        points = np.stack([x, y, z], axis=1)

        # Add colors if image provided
        if image is not None:
            image_array = np.array(image)
            colors = image_array.reshape(-1, 3) / 255.0
            points = np.concatenate([points, colors], axis=1)

        # Remove invalid points
        valid = ~np.isnan(points).any(axis=1)
        points = points[valid]

        return points
