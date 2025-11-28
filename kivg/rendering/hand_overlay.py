"""
Hand overlay functionality for whiteboard-style drawing animation.
Overlays a hand image that follows the stroke during animation.
"""
import cv2
import numpy as np
import os
from typing import Tuple, Optional


class HandOverlay:
    """Handles loading and overlaying hand images for drawing animation."""
    
    # Default hand image path (relative to package)
    DEFAULT_HAND_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "assets", "drawing-hand.png"
    )
    
    def __init__(self, hand_image_path: Optional[str] = None, 
                 scale: float = 0.30,
                 offset: Tuple[int, int] = (-18, -140)):
        """
        Initialize the hand overlay.
        
        Args:
            hand_image_path: Path to hand image (PNG with transparency).
                            If None, uses default hand image.
            scale: Scale factor for the hand image (0.0-1.0)
            offset: Offset (x, y) from the drawing point to position the hand tip
        """
        self._hand_image: Optional[np.ndarray] = None
        self._original_hand: Optional[np.ndarray] = None
        self._scale = scale
        self._offset = offset
        
        # Load hand image
        image_path = hand_image_path or self.DEFAULT_HAND_PATH
        self._load_hand_image(image_path)
    
    def _load_hand_image(self, path: str) -> None:
        """Load the hand image from file."""
        if not os.path.exists(path):
            return
        
        # Load with alpha channel
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        
        # Validate image dimensions
        if len(img.shape) < 2:
            return
        
        # Handle grayscale images (2D array)
        if len(img.shape) == 2:
            # Convert grayscale to RGBA
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif len(img.shape) == 3:
            num_channels = img.shape[2]
            if num_channels == 1:
                # Single channel, convert to RGBA
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
            elif num_channels == 3:
                # BGR to RGBA (add alpha channel)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            elif num_channels == 4:
                # BGRA to RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                # Unsupported format
                return
        else:
            # Unsupported image format
            return
        
        self._original_hand = img
        self._apply_scale()
    
    def _apply_scale(self) -> None:
        """Apply scaling to the hand image."""
        if self._original_hand is None:
            return
        
        h, w = self._original_hand.shape[:2]
        new_w = int(w * self._scale)
        new_h = int(h * self._scale)
        
        if new_w > 0 and new_h > 0:
            self._hand_image = cv2.resize(
                self._original_hand, 
                (new_w, new_h), 
                interpolation=cv2.INTER_AREA
            )
    
    @property
    def scale(self) -> float:
        """Get the current scale factor."""
        return self._scale
    
    @scale.setter
    def scale(self, value: float) -> None:
        """Set the scale factor and resize the hand image."""
        self._scale = max(0.01, min(2.0, value))  # Clamp between 0.01 and 2.0
        self._apply_scale()
    
    @property
    def offset(self) -> Tuple[int, int]:
        """Get the current offset."""
        return self._offset
    
    @offset.setter
    def offset(self, value: Tuple[int, int]) -> None:
        """Set the offset from drawing point."""
        self._offset = value
    
    @property
    def is_loaded(self) -> bool:
        """Check if hand image is loaded successfully."""
        return self._hand_image is not None
    
    def overlay_at_position(self, canvas_image: np.ndarray, 
                           x: int, y: int) -> np.ndarray:
        """
        Overlay the hand image at the specified position on the canvas.
        
        Args:
            canvas_image: The canvas image (RGBA format)
            x: X coordinate of the drawing point
            y: Y coordinate of the drawing point
            
        Returns:
            New canvas image with hand overlaid
        """
        if self._hand_image is None:
            return canvas_image
        
        # Create a copy to avoid modifying original
        result = canvas_image.copy()
        
        hand_h, hand_w = self._hand_image.shape[:2]
        canvas_h, canvas_w = result.shape[:2]
        
        # Calculate hand position (offset from drawing point)
        hand_x = x + self._offset[0]
        hand_y = y + self._offset[1]
        
        # Calculate the region to overlay
        # Handle cases where hand extends beyond canvas boundaries
        src_x1 = max(0, -hand_x)
        src_y1 = max(0, -hand_y)
        src_x2 = min(hand_w, canvas_w - hand_x)
        src_y2 = min(hand_h, canvas_h - hand_y)
        
        dst_x1 = max(0, hand_x)
        dst_y1 = max(0, hand_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Check if there's any valid region to overlay
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return result
        
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return result
        
        # Extract the regions
        hand_region = self._hand_image[src_y1:src_y2, src_x1:src_x2]
        canvas_region = result[dst_y1:dst_y2, dst_x1:dst_x2]
        
        # Perform alpha blending
        if hand_region.shape[2] == 4 and canvas_region.shape[2] == 4:
            alpha = hand_region[:, :, 3:4].astype(float) / 255.0
            
            # Blend RGB channels
            blended = (hand_region[:, :, :3].astype(float) * alpha + 
                      canvas_region[:, :, :3].astype(float) * (1 - alpha))
            
            # Blend alpha channel
            blended_alpha = (hand_region[:, :, 3:4].astype(float) + 
                           canvas_region[:, :, 3:4].astype(float) * (1 - alpha))
            
            # Combine and update result
            result[dst_y1:dst_y2, dst_x1:dst_x2, :3] = blended.astype(np.uint8)
            result[dst_y1:dst_y2, dst_x1:dst_x2, 3:4] = blended_alpha.astype(np.uint8)
        
        return result
