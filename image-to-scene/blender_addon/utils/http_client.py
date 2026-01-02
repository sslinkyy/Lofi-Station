"""
HTTP client for communicating with AI worker
"""

import requests
import base64
from typing import Optional, Dict, Any
from pathlib import Path


class WorkerClient:
    """Client for AI worker service"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check if worker is healthy"""
        response = self.session.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()

    def analyze_image(
        self,
        image_path: str,
        mode: str = "fast",
        style_preset: str = "lofi",
        reference_measurement: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Send image for analysis

        Args:
            image_path: Path to image file
            mode: Processing mode (fast/balanced/hq)
            style_preset: Style preset (lofi/realistic/architectural)
            reference_measurement: Optional scale calibration

        Returns:
            Analysis results including scene graph
        """
        # Prepare request
        files = {
            'image': open(image_path, 'rb')
        }

        settings = {
            "mode": mode,
            "style_preset": style_preset,
        }

        if reference_measurement:
            settings["reference_measurement"] = reference_measurement

        data = {
            'settings': str(settings)  # FastAPI will parse this
        }

        # Send request
        response = self.session.post(
            f"{self.base_url}/api/analyze",
            files=files,
            data=data,
            timeout=300  # 5 minutes max
        )

        response.raise_for_status()
        return response.json()

    def verify_scene(
        self,
        original_image_path: str,
        rendered_image_path: str,
        scene_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify scene by comparing renders

        Args:
            original_image_path: Path to original image
            rendered_image_path: Path to Blender render
            scene_graph: Current scene graph

        Returns:
            Verification results with suggested adjustments
        """
        files = {
            'original_image': open(original_image_path, 'rb'),
            'rendered_image': open(rendered_image_path, 'rb'),
        }

        data = {
            'scene_graph': str(scene_graph)
        }

        response = self.session.post(
            f"{self.base_url}/api/verify",
            files=files,
            data=data,
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    def close(self):
        """Close session"""
        self.session.close()
