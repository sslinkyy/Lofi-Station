"""
Vision Language Model wrappers
Supports: Qwen3-VL, FastVLM, and cloud VLMs
"""

import torch
from PIL import Image
from typing import Dict, Any, Optional, List
import logging
import json

logger = logging.getLogger(__name__)


class Qwen3VLModel:
    """
    Qwen3-VL model wrapper
    https://github.com/QwenLM/Qwen3-VL
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        quantization: str = "4bit"
    ):
        """
        Initialize Qwen3-VL model

        Args:
            model_name: Model name from Hugging Face
            device: Device (cuda/cpu)
            quantization: Quantization mode (none/4bit/8bit)
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor"""
        try:
            logger.info(f"Loading {self.model_name}...")
            from transformers import AutoModel, AutoProcessor

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with quantization
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto"
            }

            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                logger.info("Using 4-bit quantization")

            elif self.quantization == "8bit":
                load_kwargs["load_in_8bit"] = True
                logger.info("Using 8-bit quantization")

            self.model = AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            self.model.eval()
            logger.info(f"✓ {self.model_name} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Qwen3-VL: {e}")
            raise

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text from image + prompt

        Args:
            image: PIL Image
            prompt: Text prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )

            # Decode
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_scene_graph(
        self,
        image: Image.Image,
        depth_map: Optional[Image.Image] = None,
        masks: Optional[Dict[str, Any]] = None,
        style_preset: str = "lofi"
    ) -> Dict[str, Any]:
        """
        Generate scene graph from image

        Args:
            image: RGB image
            depth_map: Optional depth map image
            masks: Optional segmentation masks
            style_preset: Style (lofi/realistic/architectural)

        Returns:
            Scene graph dictionary
        """
        # Load scene graph prompt template
        from pathlib import Path
        prompt_file = Path(__file__).parent.parent / "prompts" / "scene_graph.txt"

        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                prompt_template = f.read()
        else:
            # Fallback basic prompt
            prompt_template = self._get_default_scene_graph_prompt()

        # Fill in template
        prompt = prompt_template.format(
            style_preset=style_preset,
            has_depth="yes" if depth_map else "no",
            has_masks="yes" if masks else "no"
        )

        # Generate
        response = self.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=2048,
            temperature=0.3  # Lower temp for structured output
        )

        # Parse JSON from response
        try:
            # Extract JSON from response (may have markdown code blocks)
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            scene_graph = json.loads(json_str)
            return scene_graph

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scene graph JSON: {e}")
            logger.error(f"Response was: {response}")

            # Return minimal fallback
            return {
                "error": "json_parse_failed",
                "raw_response": response,
                "camera": {"fov_deg": 50, "pitch_deg": 0, "yaw_deg": 0, "position_hint": [0, -5, 1.5]},
                "room": {"planes": []},
                "objects": [],
                "lighting": {},
                "materials": []
            }

    def _get_default_scene_graph_prompt(self) -> str:
        """Get default scene graph prompt if file not found"""
        return """You are an expert 3D scene reconstruction assistant.

Analyze this image and create a JSON scene graph.

Style: {style_preset}
Has depth map: {has_depth}
Has segmentation: {has_masks}

Output a JSON object with this structure:
{{
  "camera": {{
    "fov_deg": 45,
    "pitch_deg": -10,
    "yaw_deg": 0,
    "roll_deg": 0,
    "position_hint": [0, -4.0, 1.5]
  }},
  "room": {{
    "dimensions_m": [3.5, 4.0, 2.6],
    "planes": [
      {{"name": "floor", "normal": [0,0,1], "distance": 0.0}},
      {{"name": "back_wall", "normal": [0,1,0], "distance": 4.0}}
    ],
    "window": {{"wall": "back_wall", "rect_world": {{"x": 0.5, "y": 1.0, "width": 1.5, "height": 2.0}}}}
  }},
  "objects": [
    {{
      "id": "bed_01",
      "type": "bed",
      "bbox_px": [100, 500, 1500, 1000],
      "world_position": [1.0, 2.0, 0.5],
      "world_rotation": [0, 0, 15],
      "world_scale": [2.0, 1.6, 0.6],
      "proxy_type": "box_subdiv",
      "confidence": 0.9
    }}
  ],
  "lighting": {{
    "key_light": {{"type": "area", "position": [-1, 2, 1.5], "temp_kelvin": 2700, "strength": 100}},
    "fill_light": {{"type": "area", "position": [2, 0.5, 2], "temp_kelvin": 6500, "strength": 25}}
  }},
  "materials": [
    {{
      "target_id": "bed_01",
      "base_color": [0.8, 0.75, 0.7],
      "roughness": 0.75,
      "metallic": 0.0
    }}
  ]
}}

Output ONLY the JSON, no other text."""

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name.split("/")[-1],
            "version": "3.0",
            "device": self.device,
            "quantization": self.quantization,
            "loaded": self.model is not None,
            "capabilities": ["scene_understanding", "json_output", "multimodal"]
        }


class VLMFactory:
    """Factory for creating VLM models"""

    @staticmethod
    def create(
        model_name: str = "qwen3-vl-8b",
        device: str = "cuda",
        quantization: str = "4bit"
    ):
        """
        Create VLM model by name

        Args:
            model_name: Model identifier
            device: Device to run on
            quantization: Quantization mode

        Returns:
            VLM model instance
        """
        models = {
            "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", Qwen3VLModel),
            "qwen3-vl-4b": ("Qwen/Qwen3-VL-4B-Instruct", Qwen3VLModel),
            "qwen3-vl-2b": ("Qwen/Qwen3-VL-2B-Instruct", Qwen3VLModel),
        }

        if model_name not in models:
            raise ValueError(
                f"Unknown VLM model: {model_name}. "
                f"Available: {list(models.keys())}"
            )

        hf_name, model_class = models[model_name]
        model = model_class(
            model_name=hf_name,
            device=device,
            quantization=quantization
        )
        model.load()
        return model
