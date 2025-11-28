"""
Kivg - SVG drawing and animation using OpenCV
Core class and main API (headless, no UI required)
"""

from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np

from .core.canvas import OpenCVCanvas
from .core.animation import Animation
from .drawing.manager import DrawingManager
from .rendering.path_renderer import PathRenderer
from .rendering.shape_renderer import ShapeRenderer
from .rendering.hand_overlay import HandOverlay

# Threshold for considering fill animation complete (handles floating-point precision)
FILL_COMPLETION_THRESHOLD = 0.99


class PropertyHolder:
    """
    Simple class to hold dynamic properties for animation.
    Replaces Kivy widget for headless rendering.
    """
    
    def __init__(self, width: int = 512, height: int = 512):
        """
        Initialize the property holder.
        
        Args:
            width: Canvas width
            height: Canvas height
        """
        self.size = (width, height)
        self.pos = (0, 0)
        self.mesh_opacity = 1.0


class Kivg:
    """
    Main class for rendering and animating SVG files using OpenCV.
    
    This class provides methods to draw SVG files to an OpenCV canvas
    and generate animation frames for export.
    """

    def __init__(self, width: int = 512, height: int = 512,
                 background: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        """
        Initialize the Kivg renderer.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background: RGBA background color (0-255 for each component)
        """
        self.width = width
        self.height = height
        self.canvas = OpenCVCanvas(width, height, background)
        
        # Property holder (replaces Kivy widget)
        self.widget = PropertyHolder(width, height)
        
        # Default settings
        self._fill = True
        self._line_width = 1  # Thin stroke for smooth rendering
        self._line_color = (0, 0, 0, 255)  # RGBA 0-255
        self._animation_duration = 0.02
        self._previous_svg_file = ""
        
        # SVG data
        self.path = []
        self.closed_shapes = OrderedDict()
        self.svg_size = []
        self.current_svg_file = ""
        
        # Animation state
        self.all_anim = []
        self.curr_count = 0
        self.prev_shapes = []
        self.curr_shape = []
        
        # Stored animation frames
        self._frames: List[np.ndarray] = []
        
        # Hand overlay for whiteboard-style animation
        self._hand_overlay: Optional[HandOverlay] = None
        self._hand_draw_enabled = False

    def fill_up(self, shapes: List[List[float]], color: List[float]) -> None:
        """
        Fill shapes with specified color.
        
        Args:
            shapes: List of shape point lists to fill
            color: RGB or RGBA color (0-1 range for compatibility)
        """
        ShapeRenderer.render_mesh(self.canvas, self.widget, shapes, color, "mesh_opacity")

    def fill_up_shapes(self, *args) -> None:
        """Fill all shapes in the current SVG file, using line_color for transparent fills."""
        for id_, closed_paths in self.closed_shapes.items():
            color = self.closed_shapes[id_]["color"]
            # For shapes with transparent fill (fill="none" or invalid color),
            # use the line_color to fill them instead of skipping
            if len(color) >= 4 and color[3] == 0:
                # Use line_color for transparent fills
                fill_color = list(self._line_color)
                # Convert from 0-255 to 0-1 if needed:
                if all(c <= 1.0 for c in fill_color[:3]):
                    pass  # Already in 0-1 range
                else:
                    fill_color = [c / 255.0 for c in fill_color]
                self.fill_up(closed_paths[id_ + "shapes"], fill_color)
            else:
                self.fill_up(closed_paths[id_ + "shapes"], color)
    
    def fill_up_shapes_anim(self, shapes: List[Tuple[List[float], List[float]]], *args) -> None:
        """Fill shapes during animation."""
        for shape in shapes:
            color = shape[0]
            self.fill_up([shape[1]], color)

    def update_canvas(self, *args, **kwargs) -> None:
        """Update the canvas with the current drawing state."""
        # Convert line color from 0-1 to 0-255 if needed
        if all(c <= 1.0 for c in self._line_color[:3]):
            default_color = tuple(int(c * 255) for c in self._line_color)
        else:
            default_color = self._line_color
        
        PathRenderer.update_canvas(self.canvas, self.widget, self.path, 
                                  default_color, self._line_width)

    def draw(self, svg_file: str, animate: bool = False, 
             anim_type: str = "seq", *args, **kwargs) -> Optional[List[np.ndarray]]:
        """
        Draw an SVG file onto the canvas.
        
        Args:
            svg_file: Path to the SVG file
            animate: Whether to generate animation frames
            anim_type: Animation type - "seq" for sequential or "par" for parallel
            
        Keyword Args:
            fill: Whether to fill the drawing (bool)
            line_width: Width of lines (int)
            line_color: Color of lines (RGBA tuple, 0-255 or 0-1 range)
            dur: Duration of each animation step (float)
            fps: Frames per second for animation (int)
            from_shape_anim: Whether called from shape_animate (bool)
            hand_draw: Whether to show a hand drawing the strokes (bool)
            hand_image: Path to custom hand image (str, optional)
            hand_scale: Scale factor for hand image (float, default 0.30)
            hand_offset: Offset (x, y) from drawing point (tuple, default (-18, -140))
            
        Returns:
            List of animation frames if animate=True, None otherwise
        """
        # Process arguments
        fill = kwargs.get("fill", self._fill)
        line_width = kwargs.get("line_width", self._line_width)
        line_color = kwargs.get("line_color", self._line_color)
        duration = kwargs.get("dur", self._animation_duration)
        fps = kwargs.get("fps", 30)
        from_shape_anim = kwargs.get("from_shape_anim", False)
        anim_type = anim_type if anim_type in ("seq", "par") else "seq"
        
        # Hand drawing parameters
        hand_draw = kwargs.get("hand_draw", False)
        hand_image = kwargs.get("hand_image", None)
        hand_scale = kwargs.get("hand_scale", 0.30)
        hand_offset = kwargs.get("hand_offset", (-18, -140))
        
        # Update instance attributes
        self._fill = fill
        self._line_width = line_width
        self._line_color = line_color
        self._animation_duration = duration
        self.current_svg_file = svg_file
        
        # Set up hand overlay if enabled
        self._hand_draw_enabled = hand_draw and animate
        if self._hand_draw_enabled:
            self._hand_overlay = HandOverlay(
                hand_image_path=hand_image,
                scale=hand_scale,
                offset=hand_offset
            )
        else:
            self._hand_overlay = None
        
        # Process SVG if different from previous
        if svg_file != self._previous_svg_file:
            self.svg_size, self.closed_shapes, self.path = DrawingManager.process_path_data(svg_file)
            self._previous_svg_file = svg_file
        
        # Calculate paths
        anim_list = DrawingManager.calculate_paths(
            self.widget, self.closed_shapes, self.svg_size, 
            svg_file, animate, line_width, duration
        )
        
        # Handle rendering
        if not from_shape_anim:
            if animate:
                # Generate animation frames
                frames = self._generate_animation_frames(anim_list, fill, fps, anim_type)
                self._frames = frames
                return frames
            else:
                # Static rendering
                Animation.cancel_all(self.widget)
                self.canvas.clear()
                
                if fill:
                    # Draw fills first
                    self.fill_up_shapes()
                    # Then draw strokes on top
                    self.update_canvas()
                else:
                    # Draw only strokes
                    self.update_canvas()
                
                return None
        
        return None

    def _generate_animation_frames(self, anim_list: List[Animation], 
                                   fill: bool, fps: int, 
                                   anim_type: str) -> List[np.ndarray]:
        """
        Generate animation frames for export.
        
        Args:
            anim_list: List of animations
            fill: Whether to fill shapes
            fps: Frames per second
            anim_type: Animation type ("seq" or "par")
            
        Returns:
            List of frame images as numpy arrays
        """
        frames = []
        
        if not anim_list:
            # No animations, just return current canvas
            self.canvas.clear()
            if fill:
                self.fill_up_shapes()
            else:
                self.update_canvas()
            frames.append(self.canvas.get_image())
            return frames
        
        # Calculate total animation duration
        if anim_type == "seq":
            total_duration = sum(anim.duration for anim in anim_list)
        else:
            total_duration = max(anim.duration for anim in anim_list)
        
        # Store the stroke animation duration for hand visibility
        stroke_duration = total_duration
        
        # Add time for fill animation if needed
        if fill:
            total_duration += 0.4
        
        num_frames = max(1, int(total_duration * fps))
        
        # Store initial values for all animated properties
        initial_values = {}
        for anim in anim_list:
            for key in anim.animated_properties:
                if key not in initial_values:
                    initial_values[key] = getattr(self.widget, key, 0)
        
        # Generate frames by interpolating animation states
        for frame_idx in range(num_frames + 1):
            progress = frame_idx / num_frames if num_frames > 0 else 1.0
            current_time = progress * total_duration
            
            # Update animation properties for current time
            current_anim_idx = self._update_animation_state(
                anim_list, current_time, anim_type, initial_values
            )
            
            # Clear and redraw
            self.canvas.clear()
            
            # Calculate fill opacity
            fill_start_time = total_duration - 0.4 if fill else total_duration
            is_during_stroke = current_time < stroke_duration
            
            if fill and current_time >= fill_start_time:
                fill_progress = (current_time - fill_start_time) / 0.4
                self.widget.mesh_opacity = min(1.0, fill_progress)
                self.fill_up_shapes()
                # Draw strokes on top of fills only while fill is transitioning
                # Once fill is complete, remove strokes to show original image
                if fill_progress < FILL_COMPLETION_THRESHOLD:
                    self.update_canvas()
            elif fill:
                self.widget.mesh_opacity = 0.0
                # Draw strokes during animation
                self.update_canvas()
            else:
                self.update_canvas()
            
            # Get the current frame image
            frame_image = self.canvas.get_image()
            
            # Add hand overlay if enabled and we're during stroke animation
            if (self._hand_draw_enabled and self._hand_overlay is not None 
                and self._hand_overlay.is_loaded and is_during_stroke):
                # Get the current drawing position from the active animation
                hand_pos = self._get_current_drawing_position(
                    anim_list, current_anim_idx, anim_type
                )
                if hand_pos is not None:
                    frame_image = self._hand_overlay.overlay_at_position(
                        frame_image, hand_pos[0], hand_pos[1]
                    )
            
            frames.append(frame_image)
        
        return frames
    
    def _get_current_drawing_position(self, anim_list: List[Animation],
                                      current_anim_idx: int,
                                      anim_type: str) -> Optional[Tuple[int, int]]:
        """
        Get the current drawing position (tip of the stroke being drawn).
        
        For lines: returns the current animated end point (where the line is being drawn to)
        For beziers: calculates the point ON the bezier curve based on animation progress
        
        Args:
            anim_list: List of animations
            current_anim_idx: Index of the currently active animation
            anim_type: Animation type ("seq" or "par")
            
        Returns:
            Tuple (x, y) of the current drawing position, or None
        """
        if not anim_list or current_anim_idx < 0:
            return None
        
        if anim_type == "seq":
            # For sequential animation, get the current drawing point
            if current_anim_idx >= len(anim_list):
                return None
            
            anim = anim_list[current_anim_idx]
            props = anim.animated_properties
            
            # Check for line - for lines, the end point IS the drawing point
            for key in props:
                if key.startswith("line") and key.endswith("_end_x"):
                    prefix = key[:-6]  # Remove "_end_x"
                    # For lines, the animated end point is where we're drawing
                    x = getattr(self.widget, f"{prefix}_end_x", None)
                    y = getattr(self.widget, f"{prefix}_end_y", None)
                    if x is not None and y is not None:
                        return (int(x), int(y))
            
            # Check for bezier - need to calculate point ON the curve
            for key in props:
                if key.startswith("bezier") and key.endswith("_end_x"):
                    prefix = key[:-6]  # Remove "_end_x"
                    
                    # Get the start point (fixed)
                    start_x = getattr(self.widget, f"{prefix}_start_x", None)
                    start_y = getattr(self.widget, f"{prefix}_start_y", None)
                    
                    # Get the current animated control and end points
                    ctrl1_x = getattr(self.widget, f"{prefix}_control1_x", None)
                    ctrl1_y = getattr(self.widget, f"{prefix}_control1_y", None)
                    ctrl2_x = getattr(self.widget, f"{prefix}_control2_x", None)
                    ctrl2_y = getattr(self.widget, f"{prefix}_control2_y", None)
                    end_x = getattr(self.widget, f"{prefix}_end_x", None)
                    end_y = getattr(self.widget, f"{prefix}_end_y", None)
                    
                    if all(v is not None for v in [start_x, start_y, end_x, end_y]):
                        # Calculate progress based on how far the end point has moved
                        # The animation moves end from start to target
                        target_end_x = anim.animated_properties.get(f"{prefix}_end_x", end_x)
                        target_end_y = anim.animated_properties.get(f"{prefix}_end_y", end_y)
                        
                        # Calculate the progress (0 to 1) of this animation
                        dx = target_end_x - start_x
                        dy = target_end_y - start_y
                        
                        if abs(dx) > 0.001 or abs(dy) > 0.001:
                            # Use distance-based progress
                            total_dist = ((dx) ** 2 + (dy) ** 2) ** 0.5
                            current_dist = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
                            t = min(1.0, current_dist / total_dist) if total_dist > 0 else 1.0
                        else:
                            t = 1.0
                        
                        # Calculate point on cubic bezier at parameter t
                        # B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
                        if ctrl1_x is not None and ctrl2_x is not None:
                            # Use actual bezier calculation
                            t2 = t * t
                            t3 = t2 * t
                            mt = 1 - t
                            mt2 = mt * mt
                            mt3 = mt2 * mt
                            
                            x = mt3 * start_x + 3 * mt2 * t * ctrl1_x + 3 * mt * t2 * ctrl2_x + t3 * target_end_x
                            y = mt3 * start_y + 3 * mt2 * t * ctrl1_y + 3 * mt * t2 * ctrl2_y + t3 * target_end_y
                        else:
                            # Fallback to linear interpolation
                            x = start_x + t * (target_end_x - start_x)
                            y = start_y + t * (target_end_y - start_y)
                        
                        return (int(x), int(y))
        else:
            # For parallel animation, find the most active animation
            if anim_list:
                anim = anim_list[-1]  # Use the last animation
                props = anim.animated_properties
                for key in props:
                    if key.endswith("_end_x"):
                        prefix = key[:-6]
                        x = getattr(self.widget, f"{prefix}_end_x", None)
                        y = getattr(self.widget, f"{prefix}_end_y", None)
                        if x is not None and y is not None:
                            return (int(x), int(y))
        
        return None

    def _update_animation_state(self, anim_list: List[Animation], 
                                current_time: float, anim_type: str,
                                initial_values: Dict[str, Any]) -> int:
        """Update widget properties based on animation state at current time.
        
        Args:
            anim_list: List of animations
            current_time: Current time in the animation
            anim_type: Animation type ("seq" or "par")
            initial_values: Dictionary of initial property values
            
        Returns:
            Index of the currently active animation (-1 if none)
        """
        current_anim_idx = -1
        
        if anim_type == "seq":
            # Sequential: run animations one after another
            elapsed = 0
            # Track the end values of completed animations
            completed_values = dict(initial_values)
            
            for idx, anim in enumerate(anim_list):
                anim_end_time = elapsed + anim.duration
                
                if current_time < elapsed:
                    # Animation hasn't started yet, use completed values
                    break
                elif current_time >= elapsed and current_time < anim_end_time:
                    # This animation is active
                    current_anim_idx = idx
                    local_progress = (current_time - elapsed) / anim.duration if anim.duration > 0 else 1.0
                    local_progress = min(1.0, max(0.0, local_progress))
                    t = anim._transition(local_progress)
                    
                    for key, target in anim.animated_properties.items():
                        start_val = completed_values.get(key, initial_values.get(key, 0))
                        value = start_val + (target - start_val) * t
                        setattr(self.widget, key, value)
                    break
                else:
                    # Animation completed, update completed values
                    for key, target in anim.animated_properties.items():
                        completed_values[key] = target
                        setattr(self.widget, key, target)
                
                elapsed = anim_end_time
        else:
            # Parallel: run all animations at once
            for idx, anim in enumerate(anim_list):
                progress = current_time / anim.duration if anim.duration > 0 else 1.0
                progress = min(1.0, max(0.0, progress))
                if progress < 1.0:
                    current_anim_idx = idx
                t = anim._transition(progress)
                
                for key, target in anim.animated_properties.items():
                    start_val = initial_values.get(key, 0)
                    value = start_val + (target - start_val) * t
                    setattr(self.widget, key, value)
        
        return current_anim_idx

    def save_image(self, path: str) -> None:
        """
        Save the current canvas to an image file.
        
        Args:
            path: Output file path (.png, .jpg, etc.)
        """
        self.canvas.save(path)
    
    def save_animation(self, path: str, fps: int = 30) -> bool:
        """
        Save animation frames to a video file.
        
        Args:
            path: Output file path (.mp4, .avi, etc.)
            fps: Frames per second
            
        Returns:
            True if successful
        """
        from .export.video import write_video
        
        if not self._frames:
            return False
        
        return write_video(self._frames, path, fps)
    
    def save_gif(self, path: str, fps: int = 30) -> bool:
        """
        Save animation frames to a GIF file.
        
        Args:
            path: Output file path (.gif)
            fps: Frames per second
            
        Returns:
            True if successful
        """
        from .export.gif import save_gif
        
        if not self._frames:
            return False
        
        return save_gif(self._frames, path, fps)
    
    def get_image(self) -> np.ndarray:
        """
        Get the current canvas as a numpy array.
        
        Returns:
            Canvas image as RGBA numpy array
        """
        return self.canvas.get_image()
    
    def get_frames(self) -> List[np.ndarray]:
        """
        Get the stored animation frames.
        
        Returns:
            List of frame images as numpy arrays
        """
        return self._frames.copy()
    
    def clear(self) -> None:
        """Clear the canvas."""
        self.canvas.clear()
        self._frames = []
    
    def draw_text(self, text: str, x: float = 0, y: float = 0, 
                  animate: bool = False, **kwargs) -> Optional[List[np.ndarray]]:
        """
        Draw text on the canvas with optional animation.
        
        Args:
            text: The text to draw
            x: X position (in pixels)
            y: Y position (in pixels)
            animate: Whether to animate the text (character-by-character reveal)
            
        Keyword Args:
            font_family: Font family name (default: 'sans-serif')
            font_size: Font size in pixels (default: 32)
            font_weight: Font weight ('normal', 'bold', or numeric 100-900)
            font_style: Font style ('normal', 'italic', 'oblique')
            fill: Fill color as RGBA tuple (0-255) or hex string
            stroke: Stroke color as RGBA tuple (0-255) or hex string
            stroke_width: Stroke width in pixels
            text_anchor: Text alignment ('start', 'middle', 'end')
            letter_spacing: Space between letters in pixels
            text_decoration: Decoration ('none', 'underline', 'line-through', 'overline')
            opacity: Opacity (0-1)
            fps: Frames per second for animation (default: 30)
            duration: Total animation duration in seconds (default: 1.0)
            hand_draw: Whether to show a hand drawing the text (bool)
            hand_image: Path to custom hand image (str, optional)
            hand_scale: Scale factor for hand image (float, default 0.30)
            hand_offset: Offset (x, y) from drawing point (tuple, default (-15, -140))
            
        Returns:
            List of animation frames if animate=True, None otherwise
        """
        from .rendering.text_renderer import TextRenderer
        from .color_utils import hex_to_rgba, color_to_0_1_range
        
        # Parse color arguments
        fill_arg = kwargs.get('fill', (0, 0, 0, 255))
        if isinstance(fill_arg, str):
            fill_rgba = hex_to_rgba(fill_arg)
            fill_color = color_to_0_1_range(fill_rgba)
        else:
            # Assume it's already an RGBA tuple - check length first
            if len(fill_arg) < 3:
                fill_color = [0, 0, 0, 1.0]  # Default black
            elif all(c <= 1.0 for c in fill_arg[:3]):
                fill_color = list(fill_arg)
                if len(fill_color) == 3:
                    fill_color.append(1.0)
            else:
                fill_color = [c / 255.0 for c in fill_arg[:3]]
                alpha = fill_arg[3] / 255.0 if len(fill_arg) > 3 else 1.0
                fill_color.append(alpha)
        
        stroke_arg = kwargs.get('stroke', None)
        stroke_color = None
        if stroke_arg:
            if isinstance(stroke_arg, str):
                stroke_rgba = hex_to_rgba(stroke_arg)
                stroke_color = color_to_0_1_range(stroke_rgba)
            else:
                # Check length first to avoid IndexError
                if len(stroke_arg) < 3:
                    stroke_color = None  # Invalid color
                elif all(c <= 1.0 for c in stroke_arg[:3]):
                    stroke_color = list(stroke_arg)
                    if len(stroke_color) == 3:
                        stroke_color.append(1.0)
                else:
                    stroke_color = [c / 255.0 for c in stroke_arg[:3]]
                    alpha = stroke_arg[3] / 255.0 if len(stroke_arg) > 3 else 1.0
                    stroke_color.append(alpha)
        
        # Build text data dictionary
        text_data = {
            'text': text,
            'x': x,
            'y': y,
            'font_family': kwargs.get('font_family', 'sans-serif'),
            'font_size': kwargs.get('font_size', 32),
            'font_weight': kwargs.get('font_weight', 'normal'),
            'font_style': kwargs.get('font_style', 'normal'),
            'fill': fill_color,
            'stroke': stroke_color,
            'stroke_width': kwargs.get('stroke_width', None),
            'text_anchor': kwargs.get('text_anchor', 'start'),
            'dominant_baseline': kwargs.get('dominant_baseline', 'auto'),
            'letter_spacing': kwargs.get('letter_spacing', 0),
            'text_decoration': kwargs.get('text_decoration', 'none'),
            'opacity': kwargs.get('opacity', 1.0),
            'id': 'direct_text'
        }
        
        fps = kwargs.get('fps', 30)
        duration = kwargs.get('duration', 1.0)
        
        # Hand drawing parameters
        hand_draw = kwargs.get('hand_draw', False)
        hand_image = kwargs.get('hand_image', None)
        hand_scale = kwargs.get('hand_scale', 0.30)
        hand_offset = kwargs.get('hand_offset', (-15, -140))
        
        if animate:
            # Set up hand overlay if enabled
            hand_overlay = None
            if hand_draw:
                hand_overlay = HandOverlay(
                    hand_image_path=hand_image,
                    scale=hand_scale,
                    offset=hand_offset
                )
            
            # Generate character-by-character animation frames
            frames = self._generate_text_animation_frames(
                text_data, fps, duration, hand_overlay
            )
            self._frames = frames
            return frames
        else:
            # Static rendering
            TextRenderer.draw_text(
                self.canvas, text_data,
                scale_x=1.0, scale_y=1.0,
                offset_x=0, offset_y=0,
                opacity=1.0,
                char_reveal=-1
            )
            return None
    
    def draw_text_svg(self, svg_file: str, animate: bool = False, 
                      **kwargs) -> Optional[List[np.ndarray]]:
        """
        Draw text elements from an SVG file with optional animation.
        
        Args:
            svg_file: Path to the SVG file containing text elements
            animate: Whether to animate the text (character-by-character reveal)
            
        Keyword Args:
            fps: Frames per second for animation (default: 30)
            duration: Total animation duration in seconds (default: 2.0)
            anim_type: Animation type - 'seq' for sequential (one text at a time),
                       'par' for parallel (all texts together)
            hand_draw: Whether to show a hand drawing the text (bool)
            hand_image: Path to custom hand image (str, optional)
            hand_scale: Scale factor for hand image (float, default 0.30)
            hand_offset: Offset (x, y) from drawing point (tuple, default (-15, -140))
            
        Returns:
            List of animation frames if animate=True, None otherwise
        """
        from .svg_parser import parse_text_elements
        from .rendering.text_renderer import TextRenderer
        
        fps = kwargs.get('fps', 30)
        duration = kwargs.get('duration', 2.0)
        anim_type = kwargs.get('anim_type', 'seq')
        
        # Hand drawing parameters
        hand_draw = kwargs.get('hand_draw', False)
        hand_image = kwargs.get('hand_image', None)
        hand_scale = kwargs.get('hand_scale', 0.30)
        hand_offset = kwargs.get('hand_offset', (-15, -140))
        
        # Parse text elements from SVG
        svg_size, text_elements = parse_text_elements(svg_file)
        
        if not text_elements:
            return None
        
        # Calculate scale factors
        scale_x = self.width / svg_size[0] if svg_size[0] > 0 else 1.0
        scale_y = self.height / svg_size[1] if svg_size[1] > 0 else 1.0
        
        if animate:
            # Set up hand overlay if enabled
            hand_overlay = None
            if hand_draw:
                hand_overlay = HandOverlay(
                    hand_image_path=hand_image,
                    scale=hand_scale,
                    offset=hand_offset
                )
            
            frames = self._generate_svg_text_animation_frames(
                text_elements, scale_x, scale_y, fps, duration, anim_type, hand_overlay
            )
            self._frames = frames
            return frames
        else:
            # Static rendering - draw all text elements
            for text_data in text_elements:
                TextRenderer.draw_text(
                    self.canvas, text_data,
                    scale_x=scale_x, scale_y=scale_y,
                    offset_x=0, offset_y=0,
                    opacity=1.0,
                    char_reveal=-1
                )
            return None
    
    def _generate_text_animation_frames(self, text_data: Dict[str, Any],
                                         fps: int, duration: float,
                                         hand_overlay: Optional[HandOverlay] = None) -> List[np.ndarray]:
        """
        Generate animation frames for a single text element.
        
        Args:
            text_data: Text element data dictionary
            fps: Frames per second
            duration: Total animation duration
            hand_overlay: Optional HandOverlay instance for hand drawing effect
            
        Returns:
            List of frame images as numpy arrays
        """
        from .rendering.text_renderer import TextRenderer
        
        frames = []
        text = text_data['text']
        total_chars = len(text)
        num_frames = max(1, int(duration * fps))
        
        for frame_idx in range(num_frames + 1):
            progress = frame_idx / num_frames if num_frames > 0 else 1.0
            char_reveal = int(progress * total_chars)
            
            # Clear and redraw
            self.canvas.clear()
            
            TextRenderer.draw_text(
                self.canvas, text_data,
                scale_x=1.0, scale_y=1.0,
                offset_x=0, offset_y=0,
                opacity=1.0,
                char_reveal=char_reveal
            )
            
            # Get the current frame image
            frame_image = self.canvas.get_image()
            
            # Add hand overlay if enabled and animation is in progress
            if (hand_overlay is not None and hand_overlay.is_loaded 
                and char_reveal < total_chars):
                # Get the position of the current character being drawn
                hand_pos = TextRenderer.get_char_position(
                    text_data, char_reveal,
                    scale_x=1.0, scale_y=1.0,
                    offset_x=0, offset_y=0
                )
                if hand_pos is not None:
                    frame_image = hand_overlay.overlay_at_position(
                        frame_image, hand_pos[0], hand_pos[1]
                    )
            
            frames.append(frame_image)
        
        return frames
    
    def _generate_svg_text_animation_frames(self, text_elements: List[Dict[str, Any]],
                                             scale_x: float, scale_y: float,
                                             fps: int, duration: float,
                                             anim_type: str,
                                             hand_overlay: Optional[HandOverlay] = None) -> List[np.ndarray]:
        """
        Generate animation frames for multiple text elements from SVG.
        
        Args:
            text_elements: List of text element data dictionaries
            scale_x: X scale factor
            scale_y: Y scale factor
            fps: Frames per second
            duration: Total animation duration
            anim_type: Animation type ('seq' or 'par')
            hand_overlay: Optional HandOverlay instance for hand drawing effect
            
        Returns:
            List of frame images as numpy arrays
        """
        from .rendering.text_renderer import TextRenderer
        
        frames = []
        num_frames = max(1, int(duration * fps))
        
        if anim_type == 'seq':
            # Sequential: animate one text element at a time
            duration_per_text = duration / len(text_elements) if text_elements else duration
            
            for frame_idx in range(num_frames + 1):
                progress = frame_idx / num_frames if num_frames > 0 else 1.0
                current_time = progress * duration
                
                # Clear canvas
                self.canvas.clear()
                
                # Track the currently active text element and its char reveal for hand position
                active_text_data = None
                active_char_reveal = 0
                active_total_chars = 0
                
                for text_idx, text_data in enumerate(text_elements):
                    text_start = text_idx * duration_per_text
                    text_end = (text_idx + 1) * duration_per_text
                    text = text_data['text']
                    total_chars = len(text)
                    
                    if current_time < text_start:
                        # Text hasn't started yet
                        char_reveal = 0
                    elif current_time >= text_end:
                        # Text animation complete
                        char_reveal = total_chars
                    else:
                        # Text is animating
                        text_progress = (current_time - text_start) / duration_per_text
                        char_reveal = int(text_progress * total_chars)
                        # Track active text for hand position
                        active_text_data = text_data
                        active_char_reveal = char_reveal
                        active_total_chars = total_chars
                    
                    if char_reveal > 0:
                        TextRenderer.draw_text(
                            self.canvas, text_data,
                            scale_x=scale_x, scale_y=scale_y,
                            offset_x=0, offset_y=0,
                            opacity=1.0,
                            char_reveal=char_reveal
                        )
                
                # Get the current frame image
                frame_image = self.canvas.get_image()
                
                # Add hand overlay if enabled and animation is in progress
                if (hand_overlay is not None and hand_overlay.is_loaded 
                    and active_text_data is not None 
                    and active_char_reveal < active_total_chars):
                    # Get the position of the current character being drawn
                    hand_pos = TextRenderer.get_char_position(
                        active_text_data, active_char_reveal,
                        scale_x=scale_x, scale_y=scale_y,
                        offset_x=0, offset_y=0
                    )
                    if hand_pos is not None:
                        frame_image = hand_overlay.overlay_at_position(
                            frame_image, hand_pos[0], hand_pos[1]
                        )
                
                frames.append(frame_image)
        else:
            # Parallel: animate all text elements together
            for frame_idx in range(num_frames + 1):
                progress = frame_idx / num_frames if num_frames > 0 else 1.0
                
                # Clear canvas
                self.canvas.clear()
                
                # Track the last active text element for hand position
                last_active_text_data = None
                last_active_char_reveal = 0
                last_total_chars = 0
                
                for text_data in text_elements:
                    text = text_data['text']
                    total_chars = len(text)
                    char_reveal = int(progress * total_chars)
                    
                    if char_reveal > 0:
                        TextRenderer.draw_text(
                            self.canvas, text_data,
                            scale_x=scale_x, scale_y=scale_y,
                            offset_x=0, offset_y=0,
                            opacity=1.0,
                            char_reveal=char_reveal
                        )
                    
                    # Track the last text with animation in progress
                    if char_reveal < total_chars:
                        last_active_text_data = text_data
                        last_active_char_reveal = char_reveal
                        last_total_chars = total_chars
                
                # Get the current frame image
                frame_image = self.canvas.get_image()
                
                # Add hand overlay if enabled and animation is in progress
                if (hand_overlay is not None and hand_overlay.is_loaded 
                    and last_active_text_data is not None 
                    and last_active_char_reveal < last_total_chars):
                    # Get the position of the current character being drawn
                    hand_pos = TextRenderer.get_char_position(
                        last_active_text_data, last_active_char_reveal,
                        scale_x=scale_x, scale_y=scale_y,
                        offset_x=0, offset_y=0
                    )
                    if hand_pos is not None:
                        frame_image = hand_overlay.overlay_at_position(
                            frame_image, hand_pos[0], hand_pos[1]
                        )
                
                frames.append(frame_image)
        
        return frames
