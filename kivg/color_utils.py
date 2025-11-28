"""
Color utility functions for consistent color handling across the library.
Handles conversion between different color formats (0-1, 0-255, hex).
"""
from typing import Tuple, List, Union, Optional


def normalize_color(color: Union[Tuple, List], input_range: str = 'auto') -> Tuple[int, int, int, int]:
    """
    Normalize color to RGBA format with 0-255 range.
    
    Args:
        color: Color as tuple or list (RGB or RGBA)
        input_range: 'auto' (detect), '0-1', or '0-255'
        
    Returns:
        RGBA tuple with values in 0-255 range
    """
    if not color or len(color) < 3:
        return (0, 0, 0, 255)
    
    r, g, b = color[0], color[1], color[2]
    a = color[3] if len(color) > 3 else 1.0
    
    # Auto-detect range
    if input_range == 'auto':
        # If all RGB values <= 1.0, assume 0-1 range
        if max(r, g, b) <= 1.0:
            input_range = '0-1'
        else:
            input_range = '0-255'
    
    # Convert to 0-255 range
    if input_range == '0-1':
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        a = int(a * 255) if a <= 1.0 else int(a)
    else:
        r = int(r)
        g = int(g)
        b = int(b)
        a = int(a)
    
    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    a = max(0, min(255, a))
    
    return (r, g, b, a)


def color_to_0_1_range(color: Union[Tuple, List]) -> List[float]:
    """
    Convert color to 0-1 range (Kivy style).
    
    Args:
        color: RGBA color in any range
        
    Returns:
        List of [r, g, b, a] in 0-1 range
    """
    rgba = normalize_color(color)
    return [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0]


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[int, int, int, int]:
    """
    Convert hex color string to RGBA tuple (0-255 range).
    
    Args:
        hex_color: Hex color string (e.g., '#FF0000', '#F00', 'FF0000')
        alpha: Alpha value (0.0-1.0 or 0-255)
        
    Returns:
        RGBA tuple (0-255 range)
    """
    hex_color = hex_color.lstrip('#')
    
    # Handle 3-character hex (e.g., 'F00' -> 'FF0000')
    if len(hex_color) == 3:
        hex_color = ''.join(c * 2 for c in hex_color)
    
    # Handle 4-character hex with alpha (e.g., 'F00F' -> 'FF0000FF')
    if len(hex_color) == 4:
        hex_color = ''.join(c * 2 for c in hex_color)
    
    try:
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(alpha * 255) if alpha <= 1.0 else int(alpha)
            return (r, g, b, a)
        elif len(hex_color) == 8:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16)
            return (r, g, b, a)
    except ValueError:
        pass
    
    # Default: transparent white
    return (255, 255, 255, 0)


def rgba_to_bgra(color: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Convert RGBA to BGRA (OpenCV format).
    
    Args:
        color: RGBA tuple
        
    Returns:
        BGRA tuple
    """
    return (color[2], color[1], color[0], color[3])


def apply_opacity(color: Tuple[int, int, int, int], opacity: float) -> Tuple[int, int, int, int]:
    """
    Apply additional opacity to a color.
    
    Args:
        color: RGBA color
        opacity: Opacity multiplier (0.0-1.0)
        
    Returns:
        RGBA color with modified alpha
    """
    opacity = max(0.0, min(1.0, opacity))
    new_alpha = int(color[3] * opacity)
    return (color[0], color[1], color[2], new_alpha)


def color_0_1_to_0_255(color: Union[List[float], None]) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert color from 0-1 range (SVG/Kivy style) to 0-255 RGBA tuple (OpenCV style).
    
    Args:
        color: Color as list [r, g, b, a] in 0-1 range, or None
        
    Returns:
        RGBA tuple with values in 0-255 range, or None if input is None
    """
    if color is None:
        return None
    
    if len(color) < 3:
        return None
    
    r = int(color[0] * 255)
    g = int(color[1] * 255)
    b = int(color[2] * 255)
    a = int(color[3] * 255) if len(color) > 3 else 255
    
    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    a = max(0, min(255, a))
    
    return (r, g, b, a)
