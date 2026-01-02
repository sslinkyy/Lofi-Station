"""
Validation utilities for scene graphs and other data
"""

from typing import Dict, Any, List


def validate_scene_graph(scene_graph: Dict[str, Any]) -> List[str]:
    """
    Validate scene graph structure

    Args:
        scene_graph: Scene graph dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required keys
    required_keys = ['camera', 'room', 'objects', 'lighting', 'materials']

    for key in required_keys:
        if key not in scene_graph:
            errors.append(f"Missing required key: {key}")

    # Validate camera
    if 'camera' in scene_graph:
        camera = scene_graph['camera']
        if 'fov_deg' in camera:
            fov = camera['fov_deg']
            if fov < 10 or fov > 120:
                errors.append(f"Invalid FOV: {fov} (must be 10-120)")

    # Validate objects
    if 'objects' in scene_graph:
        for i, obj in enumerate(scene_graph['objects']):
            if 'id' not in obj:
                errors.append(f"Object {i} missing 'id'")
            if 'type' not in obj:
                errors.append(f"Object {i} missing 'type'")
            if 'world_position' not in obj:
                errors.append(f"Object {i} missing 'world_position'")

            # Check for NaN or invalid values
            if 'world_position' in obj:
                pos = obj['world_position']
                if not all(isinstance(p, (int, float)) and not (p != p) for p in pos):
                    errors.append(f"Object {i} has invalid position: {pos}")

    return errors
