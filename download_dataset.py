"""
Download and prepare CIFAR-10 dataset for image similarity search.
This script downloads CIFAR-10 test set and saves images to a directory.
"""

import os
from pathlib import Path
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image
from tqdm import tqdm


def download_and_extract_cifar10(output_dir="./sample_images", max_images=1000):
    """
    Download CIFAR-10 and extract images to a directory.

    Args:
        output_dir: Directory to save extracted images
        max_images: Maximum number of images to extract (default 1000 for small dataset)
    """
    print("Downloading CIFAR-10 dataset...")

    # Download CIFAR-10 test set
    dataset = CIFAR10(
        root="./data",
        train=False,  # Use test set
        download=True
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 class names
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    print(f"\nExtracting up to {max_images} images to {output_dir}...")

    # Extract and save images
    for idx in tqdm(range(min(max_images, len(dataset)))):
        image, label = dataset[idx]
        class_name = class_names[label]

        # Create class subdirectory
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)

        # Save image
        image_path = class_dir / f"{class_name}_{idx:05d}.png"
        image.save(image_path)

    print(f"\n✓ Successfully extracted {min(max_images, len(dataset))} images")
    print(f"  Images saved to: {output_dir}")
    print(f"  Classes: {', '.join(class_names)}")

    # Print statistics
    print("\nDataset statistics:")
    for class_name in class_names:
        class_dir = output_path / class_name
        count = len(list(class_dir.glob("*.png")))
        print(f"  {class_name}: {count} images")


if __name__ == "__main__":
    download_and_extract_cifar10()
