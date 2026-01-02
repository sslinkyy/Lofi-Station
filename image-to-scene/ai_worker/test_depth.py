"""
Test depth estimation without running full server
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import DepthModelFactory
from services import DepthService


def test_depth_estimation(image_path: str):
    """Test depth estimation on a single image"""

    print("=" * 60)
    print("Testing Depth Estimation")
    print("=" * 60)

    # Check device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load image
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")

    # Load model
    print("\nLoading Depth Pro model...")
    depth_model = DepthModelFactory.create("depth-pro", device=device)
    print("✓ Model loaded")

    # Create service
    depth_service = DepthService(
        depth_model=depth_model,
        cache_dir="cache/depth",
        enable_cache=True
    )

    # Estimate depth
    print("\nEstimating depth...")
    import asyncio
    depth_result = asyncio.run(
        depth_service.estimate_depth(image, return_confidence=True)
    )

    depth_map = depth_result["depth"]
    confidence = depth_result["confidence"]

    print(f"\n✓ Depth estimated:")
    print(f"  Range: {depth_result['min_depth']:.2f} - {depth_result['max_depth']:.2f} m")
    print(f"  Mean: {depth_result['mean_depth']:.2f} m")
    print(f"  Cached: {depth_result['cached']}")

    # Visualize
    print("\nVisualizing results...")

    fig, axes = plt.subplots(1, 3 if confidence is not None else 2, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Depth map
    depth_vis = axes[1].imshow(depth_map, cmap='plasma')
    axes[1].set_title(f"Depth Map (range: {depth_map.min():.1f} - {depth_map.max():.1f}m)")
    axes[1].axis('off')
    plt.colorbar(depth_vis, ax=axes[1])

    # Confidence map
    if confidence is not None:
        conf_vis = axes[2].imshow(confidence, cmap='viridis')
        axes[2].set_title("Confidence Map")
        axes[2].axis('off')
        plt.colorbar(conf_vis, ax=axes[2])

    plt.tight_layout()

    # Save results
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"depth_test_{Path(image_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: {output_path}")

    plt.show()

    # Test point cloud generation
    print("\nGenerating point cloud...")
    points = depth_service.depth_to_pointcloud(depth_map, image=image)
    print(f"✓ Point cloud generated: {points.shape[0]} points")

    # Save point cloud (simple format)
    pc_path = output_dir / f"pointcloud_{Path(image_path).stem}.ply"
    save_ply(pc_path, points)
    print(f"✓ Point cloud saved to: {pc_path}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


def save_ply(path: Path, points: np.ndarray):
    """Save point cloud as PLY file"""

    has_color = points.shape[1] == 6

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        for point in points:
            if has_color:
                x, y, z, r, g, b = point
                f.write(f"{x} {y} {z} {int(r*255)} {int(g*255)} {int(b*255)}\n")
            else:
                x, y, z = point
                f.write(f"{x} {y} {z}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test depth estimation")
    parser.add_argument("image", help="Path to test image")

    args = parser.parse_args()

    test_depth_estimation(args.image)
