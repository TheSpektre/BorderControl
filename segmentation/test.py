import numpy as np
import torch
import cv2
import os
import time
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def create_mask_image(anns, image=None, mask_threshold=3000, alpha=0.35):
    """
    Creates a segmentation mask image without displaying it.
    
    Args:
        anns (list): List of segmentation masks
        image (np.array): Original image (optional)
        mask_threshold (int): Minimum mask area to display
        alpha (float): Mask transparency
    
    Returns:
        np.array: Resulting image with masks
    """
    if len(anns) == 0:
        return None
        
    # Sort masks by area
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    
    # Create canvas for masks
    h, w = sorted_anns[0]['segmentation'].shape
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw masks
    for ann in sorted_anns:
        if ann['area'] > mask_threshold:
            m = ann['segmentation']
            color = (np.random.randint(0, 255), 
                     np.random.randint(0, 255), 
                     np.random.randint(0, 255))
            mask_img[m] = color
    
    # Blend with original image if provided
    if image is not None:
        # Convert image to RGB (OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Blend mask with image
        result = cv2.addWeighted(image_rgb, 1 - alpha, mask_img, alpha, 0)
    else:
        result = mask_img
    
    return result

def initialize_sam(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth", device="cpu"):
    """
    Initializes SAM model.
    
    Args:
        model_type (str): Model type
        checkpoint_path (str): Path to model weights
        device (str): Computation device (cuda/cpu)
    
    Returns:
        SamAutomaticMaskGenerator: Mask generator
    """
    sys.path.append("..")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)

def main():
    # Initialize model
    mask_generator = initialize_sam()
    
    # Load image
    image_path = '../test.jpg'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image at: {image_path}")
    
    # Generate masks
    start_time = time.perf_counter()
    masks = mask_generator.generate(image)
    end_time = time.perf_counter()
    print(f"Mask generation time: {end_time - start_time:.2f} seconds")
    
    # Create and save mask images
    output_dir = '../'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mask with threshold
    mask_img_threshold = create_mask_image(masks, image, mask_threshold=3000, alpha=1.0)
    cv2.imwrite(os.path.join(output_dir, 'mask_threshold.png'), mask_img_threshold)
    
    # Create mask without threshold
    mask_img = create_mask_image(masks, image, alpha=0.35)
    cv2.imwrite(os.path.join(output_dir, 'mask.png'), mask_img)
    
    print(f"Mask images saved to {output_dir}")

if __name__ == "__main__":
    main()
