"""
Build embedding database from images using DINOv3.
Extracts features from all images and stores them for similarity search.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms as v2
from tqdm import tqdm
import pickle


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_dinov3_model(model_name="dinov3_vits16", device=None, pretrained=False):
    """
    Load DINOv3 model from local repository.

    Args:
        model_name: Model architecture to use (default: dinov3_vits16)
        device: Device to load model on (default: auto-detect)
        pretrained: Whether to load pretrained weights (default: False)

    Returns:
        model: Loaded DINOv3 model
        device: Device model is on
    """
    if device is None:
        device = get_device()

    print(f"Loading {model_name} on {device}...")

    # Load model from local dinov3 directory
    model = torch.hub.load(
        "./dinov3",
        model_name,
        source="local",
        pretrained=pretrained
    )

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, device


def get_image_transform(image_size=224):
    """
    Get image preprocessing transform for DINOv3.

    Args:
        image_size: Size to resize images to (default: 224)

    Returns:
        transform: Torchvision transform pipeline
    """
    # Standard ImageNet normalization for LVD-1689M weights
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(image_size),
        v2.ToTensor(),
        normalize
    ])

    return transform


def extract_embedding(model, image_path, transform, device):
    """
    Extract embedding for a single image.

    Args:
        model: DINOv3 model
        image_path: Path to image file
        transform: Image preprocessing transform
        device: Device to run inference on

    Returns:
        embedding: Feature vector as numpy array
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            output = model(image_tensor)

        # Convert to numpy and normalize
        embedding = output.cpu().numpy().flatten()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # L2 normalize

        return embedding

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def build_database(
    images_dir="./sample_images",
    output_file="./embeddings_database.pkl",
    model_name="dinov3_vits16",
    pretrained=False
):
    """
    Build embedding database from directory of images.

    Args:
        images_dir: Directory containing images
        output_file: Path to save embedding database
        model_name: DINOv3 model to use
        pretrained: Whether to use pretrained weights
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.rglob(f'*{ext}'))

    image_files = sorted([str(f) for f in image_files])

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Found {len(image_files)} images")

    # Load model
    model, device = load_dinov3_model(model_name, pretrained=pretrained)
    transform = get_image_transform()

    # Extract embeddings
    embeddings = []
    valid_image_paths = []

    print("Extracting embeddings...")
    for image_path in tqdm(image_files):
        embedding = extract_embedding(model, image_path, transform, device)
        if embedding is not None:
            embeddings.append(embedding)
            valid_image_paths.append(image_path)

    # Convert to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)

    # Save database
    database = {
        'embeddings': embeddings,
        'image_paths': valid_image_paths,
        'model_name': model_name,
        'embedding_dim': embeddings.shape[1]
    }

    with open(output_file, 'wb') as f:
        pickle.dump(database, f)

    print(f"\n✓ Embedding database saved to {output_file}")
    print(f"  Total images: {len(valid_image_paths)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Database size: {embeddings.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DINOv3 embedding database")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./sample_images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./embeddings_database.pkl",
        help="Output file for embedding database"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_vits16",
        help="DINOv3 model name"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights (requires weights file)"
    )

    args = parser.parse_args()

    build_database(
        images_dir=args.images_dir,
        output_file=args.output,
        model_name=args.model,
        pretrained=args.pretrained
    )
