import cv2
import numpy as np
import os
from datetime import datetime

def create_output_dir():
    """Create output directory for processed images"""
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_processed_image(image, output_dir):
    """Save processed image with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'processed_{timestamp}.jpg'
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath

def get_image_files(directory='images'):
    """Get list of image files from directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def load_image(filepath):
    """Load image from filepath"""
    return cv2.imread(filepath)

def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if max(height, width) <= max_size:
        return image
        
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height)) 