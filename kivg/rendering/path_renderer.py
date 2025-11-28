"""
Path rendering functionality for OpenCV-based SVG rendering.
"""
from typing import List, Tuple, Any
from svg.path.path import Line, CubicBezier, Close, Move

from ..core.canvas import OpenCVCanvas
from ..path_utils import get_all_points


class PathRenderer:
    """Handles rendering of SVG paths to OpenCV canvas."""
    
    @staticmethod
    def update_canvas(canvas: OpenCVCanvas, widget: Any, 
                      path_elements: List, default_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                      default_width: int = 1) -> None:
        """
        Update the canvas with the current path elements.
        
        Args:
            canvas: OpenCVCanvas to draw on
            widget: Widget containing path properties
            path_elements: List of SVG path elements
            default_color: Default RGBA color for lines without stroke attribute
            default_width: Default width for lines without stroke-width attribute
        """
        line_count = 0
        bezier_count = 0
        
        # Draw each path element
        for element in path_elements:
            if isinstance(element, Line):
                PathRenderer._draw_line(canvas, widget, line_count, default_color, default_width)
                line_count += 1
            elif isinstance(element, CubicBezier):
                PathRenderer._draw_bezier(canvas, widget, bezier_count, default_color, default_width)
                bezier_count += 1
    
    @staticmethod
    def _draw_line(canvas: OpenCVCanvas, widget: Any, 
                   line_index: int, default_color: Tuple[int, int, int, int],
                   default_width: int) -> None:
        """Draw a line element on the canvas."""
        from ..color_utils import color_0_1_to_0_255
        
        # Check if this line has been set up with properties
        # Some path elements may not have properties if they're outside Move...Close pairs
        if not hasattr(widget, f"line{line_index}_start_x"):
            return
        
        start_x = int(getattr(widget, f"line{line_index}_start_x"))
        start_y = int(getattr(widget, f"line{line_index}_start_y"))
        end_x = int(getattr(widget, f"line{line_index}_end_x"))
        end_y = int(getattr(widget, f"line{line_index}_end_y"))
        
        # Get stroke color from SVG attributes or use default
        stroke_color = getattr(widget, f"line{line_index}_stroke_color", None)
        if stroke_color is not None:
            color = color_0_1_to_0_255(stroke_color)
            if color is None:
                color = default_color
        else:
            color = default_color
        
        # Get stroke width from SVG attributes or use default/animated width
        stroke_width = getattr(widget, f"line{line_index}_stroke_width", None)
        if stroke_width is not None:
            width = int(stroke_width)
        else:
            # Fall back to animated width or default
            width = int(getattr(widget, f"line{line_index}_width", default_width))
        
        canvas.draw_line((start_x, start_y), (end_x, end_y), color, width)
    
    @staticmethod
    def _draw_bezier(canvas: OpenCVCanvas, widget: Any, 
                     bezier_index: int, default_color: Tuple[int, int, int, int],
                     default_width: int) -> None:
        """Draw a bezier curve element on the canvas."""
        from ..color_utils import color_0_1_to_0_255
        
        # Check if this bezier has been set up with properties
        # Some path elements may not have properties if they're outside Move...Close pairs
        if not hasattr(widget, f"bezier{bezier_index}_start_x"):
            return
        
        start_x = int(getattr(widget, f"bezier{bezier_index}_start_x"))
        start_y = int(getattr(widget, f"bezier{bezier_index}_start_y"))
        ctrl1_x = int(getattr(widget, f"bezier{bezier_index}_control1_x"))
        ctrl1_y = int(getattr(widget, f"bezier{bezier_index}_control1_y"))
        ctrl2_x = int(getattr(widget, f"bezier{bezier_index}_control2_x"))
        ctrl2_y = int(getattr(widget, f"bezier{bezier_index}_control2_y"))
        end_x = int(getattr(widget, f"bezier{bezier_index}_end_x"))
        end_y = int(getattr(widget, f"bezier{bezier_index}_end_y"))
        
        # Get stroke color from SVG attributes or use default
        stroke_color = getattr(widget, f"bezier{bezier_index}_stroke_color", None)
        if stroke_color is not None:
            color = color_0_1_to_0_255(stroke_color)
            if color is None:
                color = default_color
        else:
            color = default_color
        
        # Get stroke width from SVG attributes or use default/animated width
        stroke_width = getattr(widget, f"bezier{bezier_index}_stroke_width", None)
        if stroke_width is not None:
            width = int(stroke_width)
        else:
            # Fall back to animated width or default
            width = int(getattr(widget, f"bezier{bezier_index}_width", default_width))
        
        canvas.draw_bezier(
            (start_x, start_y),
            (ctrl1_x, ctrl1_y),
            (ctrl2_x, ctrl2_y),
            (end_x, end_y),
            color, width
        )
    
    @staticmethod
    def collect_shape_points(tmp_elements_lists: List, widget: Any, 
                             shape_id: str) -> List[float]:
        """
        Collect all current points for a shape during animation.
        
        Args:
            tmp_elements_lists: Path data from shape_animate
            widget: Widget containing animation properties
            shape_id: ID of the shape
            
        Returns:
            List of points representing the current shape state
        """
        shape_list = []
        line_count = 0
        bezier_count = 0

        for path_elements in tmp_elements_lists:
            for element in path_elements:
                # Collect line points
                if len(element) == 2:  # Line
                    shape_list.extend([
                        getattr(widget, f"{shape_id}_mesh_line{line_count}_start_x"),
                        getattr(widget, f"{shape_id}_mesh_line{line_count}_start_y"),
                        getattr(widget, f"{shape_id}_mesh_line{line_count}_end_x"),
                        getattr(widget, f"{shape_id}_mesh_line{line_count}_end_y")
                    ])
                    line_count += 1
                
                # Collect bezier points
                if len(element) == 4:  # Bezier
                    shape_list.extend(
                        get_all_points(
                            (getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_start_x"),
                             getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_start_y")),
                            (getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_control1_x"),
                             getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_control1_y")),
                            (getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_control2_x"),
                             getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_control2_y")),
                            (getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_end_x"),
                             getattr(widget, f"{shape_id}_mesh_bezier{bezier_count}_end_y"))
                        )
                    )
                    bezier_count += 1
        return shape_list
