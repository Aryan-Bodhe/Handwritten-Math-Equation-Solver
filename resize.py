import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Image resizing settings
IMAGE_SIZE = (32, 32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# Detect CUDA
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def resize_image(image_path):
    """Resizes an image to 32x32 and saves it."""
    try:
        img = read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0  # Load image
        img = transforms.functional.resize(img, IMAGE_SIZE)  # Resize
        save_image(img, image_path)  # Save back
        print(f"Processed: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images_cpu(image_paths):
    """Process images using multiprocessing on CPU."""
    with Pool(cpu_count()) as pool:
        pool.map(resize_image, image_paths)

def process_images_gpu(image_paths):
    """Process images using CUDA (PyTorch)."""
    for image_path in image_paths:
        try:
            img = read_image(image_path, mode=ImageReadMode.RGB).float().to(DEVICE) / 255.0
            img = transforms.functional.resize(img, IMAGE_SIZE)
            save_image(img.cpu(), image_path)  # Move to CPU before saving
            print(f"Processed (GPU): {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    root_dir = "data/math_symbol_dataset"  # Change this to your root directory
    image_paths = [str(path) for path in Path(root_dir).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]

    if USE_CUDA:
        print("Using GPU for processing...")
        process_images_gpu(image_paths)
    else:
        print("Using CPU with multiprocessing for processing...")
        process_images_cpu(image_paths)
