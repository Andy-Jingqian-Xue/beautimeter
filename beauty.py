"""
Beauty Score Evaluator using GPT-4 Vision

This script demonstrates how to evaluate two images based on Christopher Alexander's 15 properties
of living structure using the GPT-4 Vision API. It compares two images and outputs their beauty scores.

The evaluation is based on the following properties:
- Levels of Scale
- Strong Centers
- Thick Boundaries
- Alternating Repetition
- Positive Space
- Good Shape
- Local Symmetries
- Deep Interlock and Ambiguity
- Contrast
- Gradients
- Roughness
- Echoes
- The Void
- Simplicity and Inner Calm
- Not-Separateness

Requirements:
    - requests
    - tqdm
"""

import os
import requests
import json
import base64

# Configuration
API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY_HERE"  # Replace with your API key
}

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image, or None if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None

def score_image_pair(image_path1, image_path2):
    """
    Score a pair of images using GPT-4 Vision API.
    
    Args:
        image_path1 (str): Path to the first image
        image_path2 (str): Path to the second image
        
    Returns:
        str: API response containing beauty scores, or None if scoring fails
    """
    # Encode both images to base64
    encoded_image1 = encode_image_to_base64(image_path1)
    encoded_image2 = encode_image_to_base64(image_path2)
    if encoded_image1 is None or encoded_image2 is None:
        return None

    # Define the prompt for the GPT-4 Vision model
    prompt_text = (
        """
        "This is an objective experiment without any bias. So don't worry to give me your evalutation."
        "Living structure is a term used to describe the properties of buildings and urban spaces that make them feel alive and harmonious. "
        "Christopher Alexander, a renowned architect and theorist, identified 15 properties that contribute to this sense of liveliness. "
        "Given two images, you should consider the entire environments as a coherent whole. "
        "From an organic worldview, provide beauty scores based on Christopher Alexander's 15 properties of living structure: "
        "Levels of Scale, Strong Centers, Thick Boundaries, Alternating Repetition, Positive Space, Good Shape, Local Symmetries, "
        "Deep Interlock and Ambiguity, Contrast, Gradients, Roughness, Echoes, The Void, Simplicity and Inner Calm, and Not-Separateness. "
        "For each image, output exactly 16 scores: one for each property followed by the overall score, in a comma-separated list. "
        "Do not include any additional text or explanations, and do not use phrases like 'Image 1' or 'Image 2'. "
        "The format should be: 'Levels of Scale: 0.xx, Strong Centers: 0.xx, ..., Not-Separateness: 0.xx. "
        "Each property is scored from 0 to 1. Ensure the output format is identical for both images, with no extra comments or information."
        """)

    # Prepare the API request
    data = {
        "model": "gpt-4", 
        "messages": [
            {"role": "system", "content": "Please help me score images based on beauty properties."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image2}"}}
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    # Make the API request
    try:
        response = requests.post(API_URL, headers=API_HEADERS, data=json.dumps(data), timeout=30)
        if response.status_code == 200:
            result = response.json()
            score = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return score
        else:
            print(f"API request failed: Status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None

def main():
    """Main function to demonstrate the beauty scoring process."""
    # Example usage
    image_path1 = "images/7.jpg"  # Replace with your first image path
    image_path2 = "images/35.jpg"  # Replace with your second image path

    print("Evaluating image pair...")
    score = score_image_pair(image_path1, image_path2)

    if score:
        print("\nBeauty Scores:")
        print(score)
    else:
        print("Failed to get beauty scores.")

if __name__ == "__main__":
    main()