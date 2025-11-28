"""
OpenCVCanvas - Canvas based on OpenCV for SVG rendering.
Replaces kivy.graphics.Canvas for headless rendering.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional

from ..color_utils import normalize_color


class OpenCVCanvas:
    """Canvas based on OpenCV to replace kivy.graphics.Canvas"""
    
    def __init__(self, width: int, height: int, 
                 background: Tuple[int, int, int, int] = (255, 255, 255, 255)):
        """
        Initialize the canvas.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background: RGBA background color (0-255 for each component)
        """
        self.width = width
        self.height = height
        self.background = background
        self.image = np.full((height, width, 4), background, dtype=np.uint8)
    
    def clear(self):
        """Clear the canvas to background color."""
        self.image = np.full(
            (self.height, self.width, 4), 
            self.background, 
            dtype=np.uint8
        )
    
    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                  color: Tuple[int, int, int, int], thickness: int = 1):
        """
        Draw a line on the canvas.
        
        Args:
            start: Start point (x, y)
            end: End point (x, y)
            color: RGBA color (0-255 for each component)
            thickness: Line thickness
        """
        # Normalize color
        rgba = normalize_color(color, input_range='0-255')
        
        # Handle alpha blending
        if rgba[3] < 255:
            overlay = self.image.copy()
            cv2.line(overlay, start, end, rgba, thickness, cv2.LINE_AA)
            alpha = rgba[3] / 255.0
            cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
        else:
            cv2.line(self.image, start, end, rgba, thickness, cv2.LINE_AA)
    
    def draw_polylines(self, points: np.ndarray, color: Tuple[int, int, int, int], 
                       thickness: int = 1, closed: bool = False):
        """
        Draw polylines on the canvas.
        
        Args:
            points: Array of points [(x1, y1), (x2, y2), ...]
            color: RGBA color (0-255 for each component)
            thickness: Line thickness
            closed: Whether to close the polyline
        """
        # Normalize color
        rgba = normalize_color(color, input_range='0-255')
        
        if rgba[3] < 255:
            overlay = self.image.copy()
            cv2.polylines(overlay, [points], closed, rgba, 
                         thickness, cv2.LINE_AA)
            alpha = rgba[3] / 255.0
            cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
        else:
            cv2.polylines(self.image, [points], closed, rgba, 
                         thickness, cv2.LINE_AA)
    
    def draw_bezier(self, start: Tuple[int, int], ctrl1: Tuple[int, int],
                    ctrl2: Tuple[int, int], end: Tuple[int, int],
                    color: Tuple[int, int, int, int], thickness: int = 1,
                    segments: int = 150):
        """
        Draw a cubic Bezier curve on the canvas.
        
        Args:
            start: Start point (x, y)
            ctrl1: First control point (x, y)
            ctrl2: Second control point (x, y)
            end: End point (x, y)
            color: RGBA color (0-255 for each component)
            thickness: Line thickness
            segments: Number of segments for curve approximation (higher = smoother)
        """
        points = self._calculate_bezier_points(start, ctrl1, ctrl2, end, segments)
        points = np.array(points, dtype=np.int32)
        
        self.draw_polylines(points, color, thickness, closed=False)
    
    def fill_polygon(self, points: List[Tuple[int, int]], 
                     color: Tuple[int, int, int, int]):
        """
        Fill a polygon with color.
        
        Args:
            points: List of polygon vertices [(x1, y1), (x2, y2), ...]
            color: RGBA color (0-255 for each component)
        """
        points_array = np.array(points, dtype=np.int32)
        
        # Normalize color
        rgba = normalize_color(color, input_range='0-255')
        
        if rgba[3] < 255:
            overlay = self.image.copy()
            cv2.fillPoly(overlay, [points_array], rgba)
            alpha = rgba[3] / 255.0
            cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
        else:
            cv2.fillPoly(self.image, [points_array], rgba)
    
    def fill_polygons_with_holes(self, contours: List[List[Tuple[int, int]]], 
                                  color: Tuple[int, int, int, int]):
        """
        Fill multiple polygons with holes support.
        
        When multiple contours are passed together to cv2.fillPoly, inner contours
        with opposite winding direction become holes (non-zero winding rule).
        
        Args:
            contours: List of contours, each contour is a list of (x, y) points.
                      Inner contours with opposite winding become holes.
            color: RGBA color (0-255 for each component)
        """
        if not contours:
            return
            
        # Convert all contours to numpy arrays
        contour_arrays = [np.array(c, dtype=np.int32) for c in contours if len(c) >= 3]
        
        if not contour_arrays:
            return
        
        # Normalize color
        rgba = normalize_color(color, input_range='0-255')
        
        if rgba[3] < 255:
            overlay = self.image.copy()
            cv2.fillPoly(overlay, contour_arrays, rgba)
            alpha = rgba[3] / 255.0
            cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
        else:
            cv2.fillPoly(self.image, contour_arrays, rgba)
    
    @staticmethod
    def _calculate_bezier_points(start: Tuple[float, float], ctrl1: Tuple[float, float], 
                                 ctrl2: Tuple[float, float], end: Tuple[float, float], 
                                 segments: int = 150) -> List[Tuple[int, int]]:
        """
        Calculate discrete points along a cubic Bezier curve.
        
        Uses Bernstein polynomials:
        B(t) = (1-t)³P0 + 3t(1-t)²P1 + 3t²(1-t)P2 + t³P3
        
        Args:
            start: Start point (x, y)
            ctrl1: First control point (x, y)
            ctrl2: Second control point (x, y)
            end: End point (x, y)
            segments: Number of segments (higher = smoother curves)
            
        Returns:
            List of points along the curve
        """
        points = []
        for i in range(segments + 1):
            t = i / segments
            
            # Bernstein polynomials
            b0 = (1 - t) ** 3
            b1 = 3 * t * (1 - t) ** 2
            b2 = 3 * t ** 2 * (1 - t)
            b3 = t ** 3
            
            x = b0 * start[0] + b1 * ctrl1[0] + b2 * ctrl2[0] + b3 * end[0]
            y = b0 * start[1] + b1 * ctrl1[1] + b2 * ctrl2[1] + b3 * end[1]
            
            # Use round instead of int for better precision
            points.append((round(x), round(y)))
        
        return points
    
    def save(self, path: str):
        """
        Save the canvas to an image file.
        
        Args:
            path: Output file path (.png, .jpg, etc.)
        """
        # Convert RGBA → BGR for OpenCV
        if path.lower().endswith('.png'):
            # PNG supports RGBA
            bgra = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(path, bgra)
        else:
            # JPG and others don't support alpha
            bgr = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(path, bgr)
    
    def get_image(self) -> np.ndarray:
        """
        Get the current canvas as a numpy array.
        
        Returns:
            Copy of the canvas image (RGBA format)
        """
        return self.image.copy()
    
    def get_bgr_image(self) -> np.ndarray:
        """
        Get the canvas as BGR format (for OpenCV functions).
        
        Returns:
            Canvas image in BGR format
        """
        return cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGR)
