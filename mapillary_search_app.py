"""
Gradio web UI for DINOv3-powered Mapillary image similarity search.
Search for similar Fukuoka street-level images from Mapillary.
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms as v2
import pickle
import json
from datetime import datetime


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model_and_database(
    model_name="dinov3_vits16",
    database_path="./fukuoka_embeddings.pkl",
    metadata_path="./fukuoka_images/metadata.json"
):
    """Load DINOv3 model, embedding database, and Mapillary metadata."""
    print(f"Loading model: {model_name}")

    # Load model
    device = get_device()
    model = torch.hub.load(
        "./dinov3",
        model_name,
        source="local",
        pretrained=False
    )
    model = model.to(device)
    model.eval()

    # Load embedding database
    print(f"Loading database: {database_path}")
    with open(database_path, 'rb') as f:
        database = pickle.load(f)

    embeddings = database['embeddings']
    image_paths = database['image_paths']

    # Load Mapillary metadata
    print(f"Loading metadata: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    # Create metadata lookup by local path
    metadata_dict = {m['local_path']: m for m in metadata_list}

    print(f"✓ Loaded {len(image_paths)} Fukuoka street images")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    return model, embeddings, image_paths, metadata_dict, device


def get_image_transform(image_size=224):
    """Get image preprocessing transform."""
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


def extract_query_embedding(model, image, device):
    """Extract embedding from query image."""
    transform = get_image_transform()

    # Preprocess
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        output = model(image_tensor)

    # Normalize
    embedding = output.cpu().numpy().flatten()
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

    return embedding


def compute_similarities(query_embedding, database_embeddings):
    """Compute cosine similarities between query and database."""
    # Cosine similarity
    similarities = np.dot(database_embeddings, query_embedding)
    return similarities


def format_metadata(metadata):
    """Format Mapillary metadata for display."""
    lines = []

    # Capture date
    if metadata.get('captured_at'):
        try:
            dt = datetime.fromisoformat(metadata['captured_at'].replace('Z', '+00:00'))
            lines.append(f"📅 Captured: {dt.strftime('%Y-%m-%d %H:%M')}")
        except:
            lines.append(f"📅 Captured: {metadata['captured_at']}")

    # Location
    if metadata.get('geometry'):
        coords = metadata['geometry']['coordinates']
        lines.append(f"📍 Location: {coords[1]:.6f}, {coords[0]:.6f}")

    # Compass angle
    if metadata.get('compass_angle') is not None:
        lines.append(f"🧭 Direction: {metadata['compass_angle']:.1f}°")

    # Image dimensions
    if metadata.get('width') and metadata.get('height'):
        lines.append(f"📐 Size: {metadata['width']}×{metadata['height']}")

    # Creator
    if metadata.get('creator') and metadata['creator'].get('username'):
        lines.append(f"👤 By: {metadata['creator']['username']}")

    # Mapillary ID
    if metadata.get('id'):
        mapillary_url = f"https://www.mapillary.com/app/?pKey={metadata['id']}"
        lines.append(f"🔗 [View on Mapillary]({mapillary_url})")

    return "\n".join(lines)


def search_similar_images(query_image, num_results=5):
    """
    Search for similar Fukuoka images.

    Args:
        query_image: PIL Image or numpy array
        num_results: Number of results to return

    Returns:
        results: List of (image_path, similarity_score, metadata) tuples
    """
    # Extract query embedding
    query_embedding = extract_query_embedding(
        model_global, query_image, device_global
    )

    # Compute similarities
    similarities = compute_similarities(query_embedding, embeddings_global)

    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:num_results]

    # Format results
    results = []
    for idx in top_indices:
        image_path = image_paths_global[idx]
        similarity = float(similarities[idx])
        metadata = metadata_global.get(image_path, {})
        results.append((image_path, similarity, metadata))

    return results


def gradio_search_interface(query_image, num_results=5):
    """
    Gradio interface function.

    Args:
        query_image: Uploaded image
        num_results: Number of results to return

    Returns:
        gallery_images: List of images for Gradio gallery
        results_text: Formatted results text with metadata
    """
    if query_image is None:
        return [], "Please upload an image to search."

    try:
        # Search for similar images
        results = search_similar_images(query_image, num_results)

        # Prepare gallery images and detailed text
        gallery_images = []
        results_lines = ["# 🔍 Search Results - Similar Fukuoka Street Views\n"]

        for i, (image_path, similarity, metadata) in enumerate(results, 1):
            # Add to gallery
            gallery_images.append((image_path, f"#{i}: {similarity:.4f}"))

            # Add detailed info
            results_lines.append(f"## Result {i} (Similarity: {similarity:.4f})")
            results_lines.append(format_metadata(metadata))
            results_lines.append("")  # Blank line

        results_text = "\n".join(results_lines)

        return gallery_images, results_text

    except Exception as e:
        return [], f"Error during search: {str(e)}"


def create_app():
    """Create Gradio app."""
    with gr.Blocks(title="Fukuoka Image Search - Mapillary") as app:
        gr.Markdown(
            """
            # 🗾 Fukuoka Image Similarity Search
            ### Powered by DINOv3 + Mapillary Street-Level Imagery

            Upload any image to find visually similar locations in Fukuoka, Japan.
            Search across 243 street-level images from Mapillary.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(
                    type="pil",
                    label="Upload Query Image",
                    height=400
                )

                num_results = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Results"
                )

                search_button = gr.Button(
                    "🔍 Search Similar Fukuoka Locations",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown(
                    """
                    ### 💡 Tips:
                    - Upload street-view, urban, or building photos for best results
                    - Try different types of scenes: streets, parks, landmarks
                    - Similarity scores range from 0 to 1 (higher = more similar)
                    """
                )

            with gr.Column(scale=2):
                results_gallery = gr.Gallery(
                    label="Similar Fukuoka Street Views",
                    columns=3,
                    height=500,
                    object_fit="contain"
                )

                results_text = gr.Markdown(
                    label="Detailed Results",
                    value="Upload an image and click search to see results."
                )

        # Set up search button click
        search_button.click(
            fn=gradio_search_interface,
            inputs=[query_image, num_results],
            outputs=[results_gallery, results_text]
        )

        gr.Markdown(
            """
            ---
            ### 📊 Dataset Information:
            - **Location**: Fukuoka, Japan (福岡市)
            - **Source**: Mapillary street-level imagery
            - **Images**: 243 unique viewpoints
            - **Model**: Meta DINOv3 ViT-S/16 (384-dim embeddings)

            ### 🔗 Links:
            - [Mapillary](https://www.mapillary.com/)
            - [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
            - [Fukuoka City](https://en.wikipedia.org/wiki/Fukuoka)
            """
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fukuoka Mapillary Image Search Web UI"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="./fukuoka_embeddings.pkl",
        help="Path to Fukuoka embedding database"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="./fukuoka_images/metadata.json",
        help="Path to Mapillary metadata JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_vits16",
        help="DINOv3 model name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the web UI on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )

    args = parser.parse_args()

    # Load model, database, and metadata (global for reuse)
    model_global, embeddings_global, image_paths_global, metadata_global, device_global = \
        load_model_and_database(
            model_name=args.model,
            database_path=args.database,
            metadata_path=args.metadata
        )

    # Create and launch app
    app = create_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )
