"""
Architectural Image Selector using CLIP

This script uses the CLIP (Contrastive Language-Image Pre-training) model to identify and extract
images containing architectural elements from a dataset. It processes images in batches and copies
those that meet the confidence threshold to an output directory.

Requirements:
    - torch
    - transformers
    - Pillow
    - tqdm
"""

from PIL import Image
import requests
import os
import shutil
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Configuration
INPUT_FOLDER = "samples"  # Replace with your input folder path
OUTPUT_FOLDER = "images"  # Replace with your output folder path
CONFIDENCE_THRESHOLD = 0.8  # Threshold for architectural content confidence
BATCH_SIZE = 4  # Adjust based on available GPU memory

# Labels for classification
LABELS = ["a photo with buildings", "a photo without buildings"]

class ImageDataset(Dataset):
    """Custom dataset for processing images with CLIP."""
    
    def __init__(self, input_folder, output_folder, processor):
        """
        Initialize the dataset.
        
        Args:
            input_folder (str): Path to input images
            output_folder (str): Path to save selected images
            processor: CLIP processor for image preprocessing
        """
        self.processor = processor
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_files = [
            f for f in os.listdir(input_folder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            and not os.path.exists(os.path.join(output_folder, f))
        ]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single image and its metadata."""
        filename = self.image_files[idx]
        image_path = os.path.join(self.input_folder, filename)
        try:
            image = Image.open(image_path).convert("RGB")
            return {
                "image": image,
                "filename": filename,
                "image_path": image_path
            }
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return {
                "image": Image.new("RGB", (224, 224), color="white"),
                "filename": filename,
                "image_path": image_path,
                "error": True
            }

def process_batch(batch, model, processor, labels, input_folder, output_folder, confidence_threshold):
    """
    Process a batch of images using CLIP model.
    
    Args:
        batch (dict): Batch of images and metadata
        model: CLIP model
        processor: CLIP processor
        labels (list): Classification labels
        input_folder (str): Input folder path
        output_folder (str): Output folder path
        confidence_threshold (float): Threshold for classification confidence
    """
    if not batch:
        return
    
    images = batch["image"]
    filenames = batch["filename"]
    image_paths = batch["image_path"]
    errors = batch["error"]
    
    try:
        inputs = processor(text=labels, images=images, return_tensors="pt", padding=True)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        for i, (filename, image_path, error) in enumerate(zip(filenames, image_paths, errors)):
            if error:
                continue
                
            if probs[i][0] > confidence_threshold:
                output_path = os.path.join(output_folder, filename)
                shutil.copy(image_path, output_path)
    
    except Exception as e:
        print(f"Error processing batch: {e}")

def custom_collate(batch):
    """Custom collate function for the DataLoader."""
    images = [item["image"] for item in batch]
    filenames = [item["filename"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    errors = [item.get("error", False) for item in batch]
    
    return {
        "image": images,
        "filename": filenames,
        "image_path": image_paths,
        "error": errors
    }

def main():
    """Main function to run the architectural image selection process."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Initialize dataset and dataloader
    dataset = ImageDataset(INPUT_FOLDER, OUTPUT_FOLDER, processor)
    
    if len(dataset) == 0:
        print("No images to process. Exiting...")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate
    )
    
    print(f"Processing {len(dataset)} images with batch size: {BATCH_SIZE}")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process all batches
    for batch_data in tqdm(dataloader, desc="Processing batches"):
        process_batch(
            batch_data, 
            model, 
            processor, 
            LABELS, 
            INPUT_FOLDER, 
            OUTPUT_FOLDER, 
            CONFIDENCE_THRESHOLD
        )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()