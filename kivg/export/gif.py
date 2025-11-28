"""
GIF export functionality.
"""
import cv2
import numpy as np
from typing import List, Optional

# Try to import imageio for GIF support
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def save_gif(frames: List[np.ndarray], path: str, fps: int = 30,
             loop: int = 0, quality: int = 10) -> bool:
    """
    Save a list of frames as an animated GIF.
    
    Args:
        frames: List of image arrays (RGBA format)
        path: Output file path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        quality: Image quality for GIF (1-10, higher = better quality)
        
    Returns:
        True if successful
    """
    if not HAS_IMAGEIO:
        raise ImportError(
            "imageio is required for GIF export. "
            "Install it with: pip install imageio"
        )
    
    if not frames:
        return False
    
    try:
        # Convert RGBA to RGB for GIF
        rgb_frames = []
        for frame in frames:
            if frame.shape[2] == 4:
                # Convert RGBA to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb)
        
        # Calculate duration per frame in seconds
        duration = 1.0 / fps
        
        # Save as GIF with better quality settings
        imageio.mimsave(
            path, 
            rgb_frames, 
            duration=duration, 
            loop=loop,
            quantizer='nq',  # Neural-network quantizer for better quality
            palettesize=256  # Maximum palette size
        )
        return True
    except Exception:
        # Fallback to simple save if advanced options fail
        try:
            duration = 1.0 / fps
            imageio.mimsave(path, rgb_frames, duration=duration, loop=loop)
            return True
        except Exception:
            return False


def has_gif_support() -> bool:
    """Check if GIF export is available."""
    return HAS_IMAGEIO
