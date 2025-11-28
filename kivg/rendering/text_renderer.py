"""
Text rendering functionality for OpenCV-based SVG rendering.
Handles drawing text elements with animation support.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from ..core.canvas import OpenCVCanvas


class TextRenderer:
    """Handles rendering of SVG text elements to OpenCV canvas."""
    
    # Map SVG font weights to OpenCV font thickness adjustments
    FONT_WEIGHT_MAP = {
        'normal': 1,
        'bold': 2,
        '100': 1,
        '200': 1,
        '300': 1,
        '400': 1,
        '500': 1,
        '600': 2,
        '700': 2,
        '800': 2,
        '900': 3
    }
    
    # Available OpenCV fonts (limited selection)
    OPENCV_FONTS = {
        'sans-serif': cv2.FONT_HERSHEY_SIMPLEX,
        'serif': cv2.FONT_HERSHEY_TRIPLEX,
        'monospace': cv2.FONT_HERSHEY_PLAIN,
        'arial': cv2.FONT_HERSHEY_SIMPLEX,
        'helvetica': cv2.FONT_HERSHEY_SIMPLEX,
        'times': cv2.FONT_HERSHEY_TRIPLEX,
        'times new roman': cv2.FONT_HERSHEY_TRIPLEX,
        'courier': cv2.FONT_HERSHEY_PLAIN,
        'courier new': cv2.FONT_HERSHEY_PLAIN
    }
    
    @staticmethod
    def get_opencv_font(font_family: str, font_style: str = 'normal') -> int:
        """
        Map SVG font family to OpenCV font.
        
        Args:
            font_family: SVG font family name
            font_style: Font style (normal, italic)
            
        Returns:
            OpenCV font constant
        """
        font_family_lower = font_family.lower().strip()
        
        # Try to find matching font
        for font_name, opencv_font in TextRenderer.OPENCV_FONTS.items():
            if font_name in font_family_lower:
                # Add italic variant if requested
                if font_style == 'italic' or font_style == 'oblique':
                    return opencv_font | cv2.FONT_ITALIC
                return opencv_font
        
        # Default to sans-serif
        base_font = cv2.FONT_HERSHEY_SIMPLEX
        if font_style == 'italic' or font_style == 'oblique':
            return base_font | cv2.FONT_ITALIC
        return base_font
    
    @staticmethod
    def calculate_font_scale(font_size: float) -> float:
        """
        Calculate OpenCV font scale from SVG font size.
        
        Args:
            font_size: SVG font size in pixels
            
        Returns:
            OpenCV font scale value
        """
        # OpenCV font scale is approximately: pixel_height = 30 * scale for HERSHEY_SIMPLEX
        # So scale = font_size / 30
        return font_size / 30.0
    
    @staticmethod
    def draw_text(canvas: OpenCVCanvas, text_data: Dict[str, Any],
                  scale_x: float = 1.0, scale_y: float = 1.0,
                  offset_x: float = 0, offset_y: float = 0,
                  opacity: float = 1.0,
                  char_reveal: int = -1) -> None:
        """
        Draw text element on the canvas.
        
        Args:
            canvas: OpenCVCanvas to draw on
            text_data: Dictionary with text properties
            scale_x: X scale factor for coordinate transformation
            scale_y: Y scale factor for coordinate transformation
            offset_x: X offset for positioning
            offset_y: Y offset for positioning
            opacity: Opacity multiplier (0-1)
            char_reveal: Number of characters to reveal (-1 for all)
        """
        text = text_data['text']
        if char_reveal >= 0:
            text = text[:char_reveal]
        
        if not text:
            return
        
        # Transform coordinates
        x = int(text_data['x'] * scale_x + offset_x)
        y = int(text_data['y'] * scale_y + offset_y)
        
        # Get font settings
        font = TextRenderer.get_opencv_font(
            text_data.get('font_family', 'sans-serif'),
            text_data.get('font_style', 'normal')
        )
        font_size = text_data.get('font_size', 16)
        font_scale = TextRenderer.calculate_font_scale(font_size * min(scale_x, scale_y))
        
        # Get font weight/thickness
        font_weight = text_data.get('font_weight', 'normal')
        thickness = TextRenderer.FONT_WEIGHT_MAP.get(str(font_weight), 1)
        
        # Calculate text size for alignment
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Adjust position based on text-anchor
        text_anchor = text_data.get('text_anchor', 'start')
        if text_anchor == 'middle':
            x -= text_width // 2
        elif text_anchor == 'end':
            x -= text_width
        
        # Adjust baseline
        dominant_baseline = text_data.get('dominant_baseline', 'auto')
        if dominant_baseline in ('middle', 'central'):
            y += text_height // 2
        elif dominant_baseline in ('hanging', 'text-before-edge'):
            y += text_height
        
        # Get colors
        fill_color = text_data.get('fill', [0, 0, 0, 1])
        stroke_color = text_data.get('stroke')
        stroke_width = text_data.get('stroke_width', 1)
        text_opacity = text_data.get('opacity', 1.0) * opacity
        
        # Apply letter spacing if specified
        letter_spacing = text_data.get('letter_spacing', 0) * scale_x
        
        # Convert colors to 0-255 range
        if fill_color:
            fill_rgba = tuple(int(c * 255) for c in fill_color[:3])
            fill_rgba = fill_rgba + (int(fill_color[3] * 255 * text_opacity),)
        else:
            fill_rgba = None
        
        if stroke_color:
            stroke_rgba = tuple(int(c * 255) for c in stroke_color[:3])
            stroke_rgba = stroke_rgba + (int(stroke_color[3] * 255 * text_opacity),)
        else:
            stroke_rgba = None
        
        # Draw text with letter spacing
        if letter_spacing > 0:
            TextRenderer._draw_text_with_spacing(
                canvas, text, x, y, font, font_scale, thickness,
                letter_spacing, fill_rgba, stroke_rgba, stroke_width,
                text_data.get('text_decoration', 'none'),
                text_height
            )
        else:
            TextRenderer._draw_text_simple(
                canvas, text, x, y, font, font_scale, thickness,
                fill_rgba, stroke_rgba, stroke_width,
                text_data.get('text_decoration', 'none'),
                text_width, text_height
            )
    
    @staticmethod
    def _draw_text_simple(canvas: OpenCVCanvas, text: str,
                          x: int, y: int, font: int, font_scale: float,
                          thickness: int, fill_rgba: Optional[Tuple],
                          stroke_rgba: Optional[Tuple], stroke_width: float,
                          text_decoration: str, text_width: int,
                          text_height: int) -> None:
        """Draw text without letter spacing."""
        # Draw stroke first (if any)
        if stroke_rgba and stroke_width:
            stroke_thickness = thickness + int(stroke_width * 2)
            if stroke_rgba[3] > 0:
                cv2.putText(
                    canvas.image, text, (x, y),
                    font, font_scale, stroke_rgba[:3],
                    stroke_thickness, cv2.LINE_AA
                )
        
        # Draw fill
        if fill_rgba and fill_rgba[3] > 0:
            cv2.putText(
                canvas.image, text, (x, y),
                font, font_scale, fill_rgba[:3],
                thickness, cv2.LINE_AA
            )
        
        # Draw text decoration
        TextRenderer._draw_decoration(
            canvas, text_decoration, x, y, text_width, text_height,
            fill_rgba, thickness
        )
    
    @staticmethod
    def _draw_text_with_spacing(canvas: OpenCVCanvas, text: str,
                                 x: int, y: int, font: int, font_scale: float,
                                 thickness: int, letter_spacing: float,
                                 fill_rgba: Optional[Tuple],
                                 stroke_rgba: Optional[Tuple],
                                 stroke_width: float, text_decoration: str,
                                 text_height: int) -> None:
        """Draw text with custom letter spacing."""
        current_x = x
        total_width = 0
        
        for char in text:
            # Get character width
            (char_width, _), _ = cv2.getTextSize(char, font, font_scale, thickness)
            
            # Draw stroke first
            if stroke_rgba and stroke_width:
                stroke_thickness = thickness + int(stroke_width * 2)
                if stroke_rgba[3] > 0:
                    cv2.putText(
                        canvas.image, char, (int(current_x), y),
                        font, font_scale, stroke_rgba[:3],
                        stroke_thickness, cv2.LINE_AA
                    )
            
            # Draw fill
            if fill_rgba and fill_rgba[3] > 0:
                cv2.putText(
                    canvas.image, char, (int(current_x), y),
                    font, font_scale, fill_rgba[:3],
                    thickness, cv2.LINE_AA
                )
            
            current_x += char_width + letter_spacing
            total_width += char_width + letter_spacing
        
        # Draw text decoration
        TextRenderer._draw_decoration(
            canvas, text_decoration, x, y, int(total_width - letter_spacing),
            text_height, fill_rgba, thickness
        )
    
    @staticmethod
    def _draw_decoration(canvas: OpenCVCanvas, decoration: str,
                         x: int, y: int, text_width: int, text_height: int,
                         color: Optional[Tuple], thickness: int) -> None:
        """Draw text decoration (underline, strikethrough, etc.)."""
        if not color or color[3] == 0 or decoration == 'none':
            return
        
        line_thickness = max(1, thickness // 2)
        
        if 'underline' in decoration:
            line_y = y + text_height // 4
            cv2.line(canvas.image, (x, line_y), (x + text_width, line_y),
                    color[:3], line_thickness, cv2.LINE_AA)
        
        if 'line-through' in decoration or 'strikethrough' in decoration:
            line_y = y - text_height // 3
            cv2.line(canvas.image, (x, line_y), (x + text_width, line_y),
                    color[:3], line_thickness, cv2.LINE_AA)
        
        if 'overline' in decoration:
            line_y = y - text_height
            cv2.line(canvas.image, (x, line_y), (x + text_width, line_y),
                    color[:3], line_thickness, cv2.LINE_AA)
    
    @staticmethod
    def get_text_bounds(text_data: Dict[str, Any],
                        scale_x: float = 1.0, scale_y: float = 1.0) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box for text element.
        
        Args:
            text_data: Dictionary with text properties
            scale_x: X scale factor
            scale_y: Y scale factor
            
        Returns:
            Tuple (x, y, width, height)
        """
        text = text_data['text']
        font = TextRenderer.get_opencv_font(
            text_data.get('font_family', 'sans-serif'),
            text_data.get('font_style', 'normal')
        )
        font_size = text_data.get('font_size', 16)
        font_scale = TextRenderer.calculate_font_scale(font_size * min(scale_x, scale_y))
        
        font_weight = text_data.get('font_weight', 'normal')
        thickness = TextRenderer.FONT_WEIGHT_MAP.get(str(font_weight), 1)
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x = int(text_data['x'] * scale_x)
        y = int(text_data['y'] * scale_y)
        
        return (x, y - text_height, text_width, text_height + baseline)
    
    @staticmethod
    def get_char_position(text_data: Dict[str, Any], char_index: int,
                          scale_x: float = 1.0, scale_y: float = 1.0,
                          offset_x: float = 0, offset_y: float = 0) -> Optional[Tuple[int, int]]:
        """
        Get the position (x, y) of a specific character in the text.
        
        Args:
            text_data: Dictionary with text properties
            char_index: Index of the character (0-based)
            scale_x: X scale factor
            scale_y: Y scale factor
            offset_x: X offset for positioning
            offset_y: Y offset for positioning
            
        Returns:
            Tuple (x, y) of the character position, or None if invalid
        """
        text = text_data['text']
        if char_index < 0 or char_index > len(text):
            return None
        
        if char_index == 0:
            # Return the starting position
            x = int(text_data['x'] * scale_x + offset_x)
            y = int(text_data['y'] * scale_y + offset_y)
            return (x, y)
        
        # Get font settings
        font = TextRenderer.get_opencv_font(
            text_data.get('font_family', 'sans-serif'),
            text_data.get('font_style', 'normal')
        )
        font_size = text_data.get('font_size', 16)
        font_scale = TextRenderer.calculate_font_scale(font_size * min(scale_x, scale_y))
        
        font_weight = text_data.get('font_weight', 'normal')
        thickness = TextRenderer.FONT_WEIGHT_MAP.get(str(font_weight), 1)
        
        # Calculate base position
        x = int(text_data['x'] * scale_x + offset_x)
        y = int(text_data['y'] * scale_y + offset_y)
        
        # Calculate full text size for alignment
        full_text = text_data['text']
        (full_width, text_height), _ = cv2.getTextSize(
            full_text, font, font_scale, thickness
        )
        
        # Adjust position based on text-anchor
        text_anchor = text_data.get('text_anchor', 'start')
        if text_anchor == 'middle':
            x -= full_width // 2
        elif text_anchor == 'end':
            x -= full_width
        
        # Adjust baseline
        dominant_baseline = text_data.get('dominant_baseline', 'auto')
        if dominant_baseline in ('middle', 'central'):
            y += text_height // 2
        elif dominant_baseline in ('hanging', 'text-before-edge'):
            y += text_height
        
        # Calculate width of text up to char_index
        letter_spacing = text_data.get('letter_spacing', 0) * scale_x
        
        if letter_spacing > 0:
            # Calculate width character by character
            current_x = x
            for i in range(min(char_index, len(text))):
                char = text[i]
                (char_width, _), _ = cv2.getTextSize(char, font, font_scale, thickness)
                current_x += char_width + letter_spacing
            x = int(current_x)
        else:
            # Calculate width of substring
            substring = text[:char_index]
            (sub_width, _), _ = cv2.getTextSize(substring, font, font_scale, thickness)
            x += sub_width
        
        return (x, y)
