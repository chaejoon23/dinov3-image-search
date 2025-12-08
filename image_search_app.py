"""
Gradio web UI for DINOv3-powered image similarity search.
Allows users to upload an image and find similar images in the database.
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms as v2
import pickle


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
    database_path="./embeddings_database.pkl"
):
    """Load DINOv3 model and embedding database."""
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

    # Load database
    print(f"Loading database: {database_path}")
    with open(database_path, 'rb') as f:
        database = pickle.load(f)

    embeddings = database['embeddings']
    image_paths = database['image_paths']

    print(f"✓ Loaded {len(image_paths)} images")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    return model, embeddings, image_paths, device


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


def search_similar_images(query_image, num_results=5):
    """
    Search for similar images.

    Args:
        query_image: PIL Image or numpy array
        num_results: Number of results to return

    Returns:
        results: List of (image_path, similarity_score) tuples
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
        results.append((image_path, similarity))

    return results


def gradio_search_interface(query_image, num_results=5):
    """
    Gradio interface function.

    Args:
        query_image: Uploaded image
        num_results: Number of results to return

    Returns:
        gallery_images: List of images for Gradio gallery
        results_text: Formatted results text
    """
    if query_image is None:
        return [], "Please upload an image to search."

    try:
        # Search for similar images
        results = search_similar_images(query_image, num_results)

        # Prepare gallery images
        gallery_images = []
        results_lines = []

        for i, (image_path, similarity) in enumerate(results, 1):
            gallery_images.append((image_path, f"#{i}: {similarity:.4f}"))
            results_lines.append(
                f"{i}. {Path(image_path).name} (similarity: {similarity:.4f})"
            )

        results_text = "\n".join(results_lines)

        return gallery_images, results_text

    except Exception as e:
        return [], f"Error during search: {str(e)}"


def create_app():
    """Create Gradio app."""
    with gr.Blocks(title="DINOv3 Image Search") as app:
        gr.Markdown(
            """
            # 🔍 DINOv3 Image Similarity Search

            Upload an image to find visually similar images in the database.
            Powered by Meta's DINOv3 vision foundation model.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(
                    type="pil",
                    label="Upload Query Image",
                    height=300
                )

                num_results = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Results"
                )

                search_button = gr.Button("🔍 Search Similar Images", variant="primary")

            with gr.Column(scale=2):
                results_gallery = gr.Gallery(
                    label="Similar Images",
                    columns=3,
                    height=400,
                    object_fit="contain"
                )

                results_text = gr.Textbox(
                    label="Results",
                    lines=10,
                    max_lines=20
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
            ### How it works:
            1. **Upload** an image you want to search for
            2. **Select** the number of similar images to retrieve
            3. **Click** the search button
            4. **View** results ranked by visual similarity

            The similarity score ranges from 0 to 1, where 1 is identical.
            """
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DINOv3 Image Search Web UI")
    parser.add_argument(
        "--database",
        type=str,
        default="./embeddings_database.pkl",
        help="Path to embedding database"
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
        default=7860,
        help="Port to run the web UI on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )

    args = parser.parse_args()

    # Load model and database (global for reuse)
    model_global, embeddings_global, image_paths_global, device_global = \
        load_model_and_database(
            model_name=args.model,
            database_path=args.database
        )

    # Create and launch app
    app = create_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )
