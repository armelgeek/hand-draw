"""
Image export functionality.
"""
import cv2
import numpy as np
from typing import Optional


def save_image(image: np.ndarray, path: str, quality: int = 95) -> bool:
    """
    Save an image to a file.
    
    Args:
        image: Image array (RGBA format)
        path: Output file path
        quality: JPEG quality (0-100) or PNG compression (0-9)
        
    Returns:
        True if save was successful
    """
    try:
        if path.lower().endswith('.png'):
            # PNG supports RGBA
            bgra = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            compression = min(9, max(0, int(9 - quality / 11)))
            cv2.imwrite(path, bgra, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        elif path.lower().endswith(('.jpg', '.jpeg')):
            # JPEG doesn't support alpha
            bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            # Default to BGR
            bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(path, bgr)
        return True
    except Exception:
        return False


def save_sequence(images: list, output_dir: str, prefix: str = "frame",
                  format: str = "png") -> list:
    """
    Save a sequence of images to files.
    
    Args:
        images: List of image arrays
        output_dir: Output directory
        prefix: Filename prefix
        format: Image format (png, jpg)
        
    Returns:
        List of saved file paths
    """
    import os
    
    paths = []
    for i, image in enumerate(images):
        filename = f"{prefix}_{i:05d}.{format}"
        path = os.path.join(output_dir, filename)
        if save_image(image, path):
            paths.append(path)
    
    return paths
