"""
Download Fukuoka street-level images from Mapillary API.
Uses DINOv3 for visual similarity search on real-world imagery.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import json
import time
from PIL import Image
from io import BytesIO
import argparse


class MapillaryDownloader:
    """Download images from Mapillary API."""

    def __init__(self, access_token, output_dir="./fukuoka_images"):
        self.access_token = access_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.json"
        self.base_url = "https://graph.mapillary.com"

        # Fukuoka city approximate boundaries
        # Center: 33.5904, 130.4017
        self.fukuoka_center = {"lat": 33.5904, "lon": 130.4017}

    def generate_grid_tiles(self, center_lat, center_lon, grid_size=0.008, num_tiles=5):
        """
        Generate grid of bounding boxes covering Fukuoka.
        Mapillary requires bbox < 0.01 degrees.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            grid_size: Size of each tile (degrees)
            num_tiles: Number of tiles in each direction

        Returns:
            List of (minLon, minLat, maxLon, maxLat) tuples
        """
        tiles = []
        offset = (num_tiles // 2) * grid_size

        for i in range(num_tiles):
            for j in range(num_tiles):
                min_lon = center_lon - offset + (i * grid_size)
                max_lon = min_lon + grid_size
                min_lat = center_lat - offset + (j * grid_size)
                max_lat = min_lat + grid_size

                tiles.append((min_lon, min_lat, max_lon, max_lat))

        return tiles

    def search_images_in_bbox(self, bbox, max_images=50):
        """
        Search for images within a bounding box.

        Args:
            bbox: (minLon, minLat, maxLon, maxLat)
            max_images: Maximum images to retrieve per bbox

        Returns:
            List of image metadata dictionaries
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        url = f"{self.base_url}/images"
        params = {
            "access_token": self.access_token,
            "fields": "id,thumb_1024_url,captured_at,compass_angle,geometry,creator,width,height",
            "bbox": bbox_str,
            "limit": max_images
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching images for bbox {bbox_str}: {e}")
            return []

    def download_image(self, image_url, image_id):
        """
        Download image from URL and save to disk.

        Args:
            image_url: URL to download image from
            image_id: Mapillary image ID

        Returns:
            Path to saved image or None if failed
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Open and save image
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Ensure RGB format

            # Save with image ID as filename
            image_path = self.output_dir / f"mapillary_{image_id}.jpg"
            img.save(image_path, "JPEG", quality=95)

            return str(image_path)

        except Exception as e:
            print(f"Error downloading image {image_id}: {e}")
            return None

    def download_fukuoka_images(self, target_count=500, images_per_tile=20):
        """
        Download images from Fukuoka area.

        Args:
            target_count: Target number of images to download
            images_per_tile: Max images per grid tile

        Returns:
            List of downloaded image metadata
        """
        print("🗾 Downloading Fukuoka street-level images from Mapillary...")
        print(f"Target: ~{target_count} images\n")

        # Generate grid tiles covering Fukuoka
        grid_size = 0.008  # ~0.8km per tile at this latitude
        num_tiles = max(5, int((target_count / images_per_tile) ** 0.5) + 1)
        tiles = self.generate_grid_tiles(
            self.fukuoka_center["lat"],
            self.fukuoka_center["lon"],
            grid_size=grid_size,
            num_tiles=num_tiles
        )

        print(f"Searching {len(tiles)} grid tiles across Fukuoka...")

        all_images = []
        downloaded_images = []

        # Search for images in each tile
        for bbox in tqdm(tiles, desc="Searching tiles"):
            images = self.search_images_in_bbox(bbox, max_images=images_per_tile)
            all_images.extend(images)
            time.sleep(0.1)  # Rate limiting

            if len(all_images) >= target_count:
                break

        # Remove duplicates by image ID
        unique_images = {img["id"]: img for img in all_images}.values()
        images_to_download = list(unique_images)[:target_count]

        print(f"\n✓ Found {len(images_to_download)} unique images")
        print(f"Downloading images...\n")

        # Download images
        for img_data in tqdm(images_to_download, desc="Downloading"):
            image_url = img_data.get("thumb_1024_url")
            image_id = img_data["id"]

            if not image_url:
                continue

            local_path = self.download_image(image_url, image_id)

            if local_path:
                # Store metadata with local path
                metadata = {
                    "id": image_id,
                    "local_path": local_path,
                    "mapillary_url": image_url,
                    "captured_at": img_data.get("captured_at"),
                    "compass_angle": img_data.get("compass_angle"),
                    "geometry": img_data.get("geometry"),
                    "creator": img_data.get("creator"),
                    "width": img_data.get("width"),
                    "height": img_data.get("height")
                }
                downloaded_images.append(metadata)

            time.sleep(0.05)  # Rate limiting

        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(downloaded_images, f, indent=2)

        print(f"\n✅ Successfully downloaded {len(downloaded_images)} images")
        print(f"📂 Images saved to: {self.output_dir}")
        print(f"📄 Metadata saved to: {self.metadata_file}")

        return downloaded_images


def main():
    parser = argparse.ArgumentParser(
        description="Download Fukuoka images from Mapillary"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Mapillary API access token (format: MLY|...)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fukuoka_images",
        help="Output directory for images"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Target number of images to download (default: 500)"
    )
    parser.add_argument(
        "--images-per-tile",
        type=int,
        default=20,
        help="Max images per grid tile (default: 20)"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = MapillaryDownloader(
        access_token=args.token,
        output_dir=args.output_dir
    )

    # Download images
    downloader.download_fukuoka_images(
        target_count=args.count,
        images_per_tile=args.images_per_tile
    )


if __name__ == "__main__":
    main()
