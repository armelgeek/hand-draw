"""
Hand Renderer for Kivg - Drawing Hand Animation Support (OpenCV-based)

This module provides the drawing hand functionality for the Kivg SVG animation library.
It renders a hand image that follows the drawing path, creating a realistic
handwriting effect.

This version uses OpenCV for image processing instead of Kivy, enabling headless
rendering without a display server.

Usage:
    from kivg.hand_rendering import HandRenderer
    
    # Initialize with hand image path
    hand_renderer = HandRenderer(
        hand_image_path='path/to/drawing-hand.png',
        hand_mask_path='path/to/hand-mask.png'
    )
    
    # Attach to a Kivg instance
    kivg_instance = Kivg(widget)
    hand_renderer.attach_to_kivg(kivg_instance)
    
    # Draw with hand animation
    kivg_instance.draw('path/to/svg', animate=True, show_hand=True)
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, List, Any, Callable
from dataclasses import dataclass


# Maximum number of path elements to search when finding drawing position
# This is a reasonable upper limit for complex SVG paths
MAX_PATH_ELEMENT_SEARCH = 100


@dataclass
class HandConfig:
    """Configuration for hand rendering."""
    hand_image_path: str
    hand_mask_path: Optional[str] = None
    scale: float = 1.0
    offset_x: int = 0  # Offset from drawing point (pen tip adjustment)
    offset_y: int = 0
    rotation: float = 0.0  # Rotation angle in degrees
    opacity: float = 1.0
    visible: bool = True


class HandRenderer:
    """
    Renders a drawing hand that follows SVG path animation using OpenCV.
    
    This class manages the hand image display and updates its position
    based on the current drawing progress in Kivg animations.
    """
    
    def __init__(self, 
                 hand_image_path: str,
                 hand_mask_path: Optional[str] = None,
                 scale: float = 0.15,
                 offset_x: int = -30,
                 offset_y: int = -10):
        """
        Initialize the HandRenderer.
        
        Args:
            hand_image_path: Path to the hand image (PNG with transparency recommended)
            hand_mask_path: Optional path to hand mask for alpha compositing
            scale: Scale factor for the hand image (default 0.15 = 15% of original size)
            offset_x: X offset from drawing point (pen tip adjustment)
            offset_y: Y offset from drawing point (pen tip adjustment)
        """
        self.config = HandConfig(
            hand_image_path=hand_image_path,
            hand_mask_path=hand_mask_path,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y
        )
        
        # OpenCV-based image storage (replaces Kivy textures)
        self._hand_image: Optional[np.ndarray] = None  # RGBA numpy array
        self._original_hand: Optional[np.ndarray] = None  # Original unscaled image
        self._mask_image: Optional[np.ndarray] = None  # Alpha mask
        self._hand_size: Tuple[int, int] = (0, 0)  # (width, height)
        self._current_pos: Tuple[float, float] = (0, 0)
        self._kivg_instance: Optional[Any] = None
        self._widget: Optional[Any] = None
        self._is_visible: bool = False
        self._original_update_canvas: Optional[Callable] = None
        self._original_track_progress: Optional[Callable] = None
        
        # Load images using OpenCV
        self._load_images()
    
    def _load_images(self) -> None:
        """Load hand and mask images from files using OpenCV."""
        if os.path.exists(self.config.hand_image_path):
            try:
                # Load with alpha channel using OpenCV
                img = cv2.imread(self.config.hand_image_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Handle different image formats
                    if len(img.shape) == 2:
                        # Grayscale to RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
                    elif len(img.shape) == 3:
                        if img.shape[2] == 3:
                            # BGR to RGBA
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                        elif img.shape[2] == 4:
                            # BGRA to RGBA
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                    
                    self._original_hand = img
                    self._apply_scale()
                else:
                    print(f"[HandRenderer] Failed to load hand image: {self.config.hand_image_path}")
            except Exception as e:
                print(f"[HandRenderer] Failed to load hand image: {e}")
        else:
            print(f"[HandRenderer] Hand image not found: {self.config.hand_image_path}")
        
        if self.config.hand_mask_path and os.path.exists(self.config.hand_mask_path):
            try:
                mask = cv2.imread(self.config.hand_mask_path, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    # Convert to grayscale if needed
                    if len(mask.shape) == 3:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    self._mask_image = mask
            except Exception as e:
                print(f"[HandRenderer] Failed to load hand mask: {e}")
    
    def _apply_scale(self) -> None:
        """Apply scaling to the hand image."""
        if self._original_hand is None:
            return
        
        h, w = self._original_hand.shape[:2]
        new_w = int(w * self.config.scale)
        new_h = int(h * self.config.scale)
        
        if new_w > 0 and new_h > 0:
            self._hand_image = cv2.resize(
                self._original_hand,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )
            self._hand_size = (new_w, new_h)
    
    def attach_to_kivg(self, kivg_instance: Any) -> 'HandRenderer':
        """
        Attach this HandRenderer to a Kivg instance.
        
        This method patches the Kivg instance to include hand rendering
        during SVG path animations.
        
        Args:
            kivg_instance: The Kivg instance to attach to
            
        Returns:
            Self for method chaining
        """
        self._kivg_instance = kivg_instance
        self._widget = kivg_instance.widget
        
        # Store original methods
        self._original_update_canvas = kivg_instance.update_canvas
        self._original_track_progress = kivg_instance.track_progress
        
        # Patch methods to include hand rendering
        kivg_instance.update_canvas = self._patched_update_canvas
        kivg_instance.track_progress = self._patched_track_progress
        
        # Add show_hand property to kivg instance
        kivg_instance._show_hand = False
        kivg_instance._hand_renderer = self
        
        return self
    
    def detach(self) -> None:
        """Detach the HandRenderer from the Kivg instance."""
        if self._kivg_instance:
            # Restore original methods
            if self._original_update_canvas:
                self._kivg_instance.update_canvas = self._original_update_canvas
            if self._original_track_progress:
                self._kivg_instance.track_progress = self._original_track_progress
            
            self._kivg_instance._show_hand = False
            self._kivg_instance._hand_renderer = None
            self._kivg_instance = None
        
        self.hide_hand()
    
    def _patched_update_canvas(self, *args, **kwargs) -> None:
        """Patched update_canvas that includes hand rendering."""
        # Call original update_canvas
        self._original_update_canvas(*args, **kwargs)
        
        # Show hand if enabled
        if getattr(self._kivg_instance, '_show_hand', False):
            self._update_hand_position_from_progress()
    
    def _patched_track_progress(self, *args) -> None:
        """Patched track_progress that includes hand rendering."""
        # Call original track_progress
        self._original_track_progress(*args)
        
        # Update hand position if enabled
        if getattr(self._kivg_instance, '_show_hand', False):
            self._update_hand_position_from_shape()
    
    def _update_hand_position_from_progress(self) -> None:
        """Update hand position based on current line/bezier endpoints."""
        if not self._widget:
            return
        
        # Try to get the current drawing position from line properties
        # This looks for the most recently animated line or bezier endpoint
        pos = self._find_current_drawing_position()
        if pos:
            self.show_hand_at(pos[0], pos[1])
    
    def _update_hand_position_from_shape(self) -> None:
        """Update hand position based on current shape animation state."""
        if not self._kivg_instance:
            return
        
        # Get current shape ID
        curr_id = getattr(self._kivg_instance, "curr_id", None)
        if not curr_id:
            return
        
        # Try to find current position from mesh properties
        pos = self._find_shape_position(curr_id)
        if pos:
            self.show_hand_at(pos[0], pos[1])
    
    def _find_current_drawing_position(self) -> Optional[Tuple[float, float]]:
        """Find the current drawing position from widget properties."""
        if not self._widget:
            return None
        
        # Search for line endpoints (most common case)
        pos = self._find_endpoint_position("line", "")
        if pos:
            return pos
        
        # Search for bezier endpoints
        return self._find_endpoint_position("bezier", "")
    
    def _find_shape_position(self, shape_id: str) -> Optional[Tuple[float, float]]:
        """Find position from shape mesh properties."""
        if not self._widget:
            return None
        
        # Search for mesh line endpoints
        pos = self._find_endpoint_position("line", f"{shape_id}_mesh_", check_width=False)
        if pos:
            return pos
        
        # Search for mesh bezier endpoints
        return self._find_endpoint_position("bezier", f"{shape_id}_mesh_", check_width=False)
    
    def _find_endpoint_position(self, element_type: str, prefix: str, 
                                 check_width: bool = True) -> Optional[Tuple[float, float]]:
        """
        Find endpoint position from widget properties.
        
        Args:
            element_type: Type of element ('line' or 'bezier')
            prefix: Property name prefix (e.g., '' for lines, 'shape_id_mesh_' for mesh lines)
            check_width: Whether to check if width > 0 (for detecting active animations)
            
        Returns:
            Tuple of (x, y) coordinates if found, None otherwise
        """
        if not self._widget:
            return None
        
        for i in range(MAX_PATH_ELEMENT_SEARCH):
            prop_prefix = f"{prefix}{element_type}{i}"
            try:
                end_x = getattr(self._widget, f"{prop_prefix}_end_x", None)
                end_y = getattr(self._widget, f"{prop_prefix}_end_y", None)
                if end_x is not None and end_y is not None:
                    if check_width:
                        width = getattr(self._widget, f"{prop_prefix}_width", 0)
                        if width > 0:
                            return (end_x, end_y)
                    else:
                        return (end_x, end_y)
            except AttributeError:
                break
        
        return None
    
    def show_hand_at(self, x: float, y: float) -> None:
        """
        Show the drawing hand at the specified position.
        
        Note: In OpenCV mode, this method just updates the position.
        The actual overlay happens in overlay_on_image().
        
        Args:
            x: X coordinate (pen tip position)
            y: Y coordinate (pen tip position)
        """
        if self._hand_image is None:
            return
        
        self._current_pos = (x, y)
        self._is_visible = True
    
    def hide_hand(self) -> None:
        """Hide the drawing hand."""
        self._is_visible = False
    
    def overlay_on_image(self, canvas_image: np.ndarray, 
                         x: Optional[float] = None, 
                         y: Optional[float] = None) -> np.ndarray:
        """
        Overlay the hand image on a canvas image using OpenCV.
        
        This is the core rendering method that composites the hand image
        onto the canvas at the specified position (or the last known position).
        
        Args:
            canvas_image: The canvas image (RGBA format)
            x: X coordinate of the drawing point (uses last position if None)
            y: Y coordinate of the drawing point (uses last position if None)
            
        Returns:
            New canvas image with hand overlaid
        """
        if self._hand_image is None or not self._is_visible:
            return canvas_image
        
        # Use provided position or fall back to last known position
        pos_x = x if x is not None else self._current_pos[0]
        pos_y = y if y is not None else self._current_pos[1]
        
        # Update current position
        self._current_pos = (pos_x, pos_y)
        
        # Create a copy to avoid modifying original
        result = canvas_image.copy()
        
        hand_h, hand_w = self._hand_image.shape[:2]
        canvas_h, canvas_w = result.shape[:2]
        
        # Calculate hand position with offset (pen tip adjustment)
        hand_x = int(pos_x + self.config.offset_x)
        hand_y = int(pos_y + self.config.offset_y)
        
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
        
        # Perform alpha blending with opacity support
        if hand_region.shape[2] == 4 and len(canvas_region.shape) == 3 and canvas_region.shape[2] >= 3:
            # Apply opacity to alpha channel
            alpha = hand_region[:, :, 3:4].astype(float) / 255.0 * self.config.opacity
            
            # Blend RGB channels using standard alpha compositing: result = src * alpha + dst * (1 - alpha)
            blended = (hand_region[:, :, :3].astype(float) * alpha + 
                      canvas_region[:, :, :3].astype(float) * (1 - alpha))
            
            # Update result
            result[dst_y1:dst_y2, dst_x1:dst_x2, :3] = blended.astype(np.uint8)
            
            # Blend alpha channel if canvas has alpha using Porter-Duff over: out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
            if canvas_region.shape[2] == 4:
                canvas_alpha = canvas_region[:, :, 3:4].astype(float) / 255.0
                blended_alpha = (alpha + canvas_alpha * (1 - alpha)) * 255.0
                result[dst_y1:dst_y2, dst_x1:dst_x2, 3:4] = blended_alpha.astype(np.uint8)
        
        return result
    
    def set_scale(self, scale: float) -> None:
        """Set the hand image scale and resize the image."""
        self.config.scale = max(0.01, min(2.0, scale))  # Clamp between 0.01 and 2.0
        self._apply_scale()
    
    def set_offset(self, offset_x: int, offset_y: int) -> None:
        """Set the pen tip offset."""
        self.config.offset_x = offset_x
        self.config.offset_y = offset_y
    
    def set_opacity(self, opacity: float) -> None:
        """Set the hand opacity (0.0 to 1.0)."""
        self.config.opacity = max(0.0, min(1.0, opacity))
    
    @property
    def is_visible(self) -> bool:
        """Check if the hand is currently visible."""
        return self._is_visible
    
    @property
    def is_loaded(self) -> bool:
        """Check if hand image is loaded successfully."""
        return self._hand_image is not None
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get the current hand position."""
        return self._current_pos
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get the current hand size (width, height)."""
        return self._hand_size
    
    @property
    def hand_image(self) -> Optional[np.ndarray]:
        """Get the scaled hand image (RGBA format)."""
        return self._hand_image


def enable_hand_drawing(kivg_instance: Any, 
                        hand_image_path: str,
                        hand_mask_path: Optional[str] = None,
                        scale: float = 0.15,
                        offset_x: int = -30,
                        offset_y: int = -10) -> HandRenderer:
    """
    Convenience function to enable hand drawing on a Kivg instance.
    
    Args:
        kivg_instance: The Kivg instance to enable hand drawing on
        hand_image_path: Path to the hand image
        hand_mask_path: Optional path to hand mask
        scale: Scale factor for the hand image
        offset_x: X offset from drawing point
        offset_y: Y offset from drawing point
        
    Returns:
        The HandRenderer instance for further configuration
        
    Example:
        from kivg import Kivg
        from kivg.hand_rendering import enable_hand_drawing
        
        kivg = Kivg(my_widget)
        hand = enable_hand_drawing(kivg, 'path/to/hand.png')
        kivg._show_hand = True  # Enable hand visibility
        kivg.draw('my_svg.svg', animate=True)
    """
    hand_renderer = HandRenderer(
        hand_image_path=hand_image_path,
        hand_mask_path=hand_mask_path,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y
    )
    hand_renderer.attach_to_kivg(kivg_instance)
    return hand_renderer