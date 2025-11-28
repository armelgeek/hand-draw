"""
SVG parsing utilities for Kivg.
Handles parsing SVG files and extracting path data and text elements.
"""
from typing import Tuple, List, Dict, Any, Optional
from xml.dom import minidom
import re

from .color_utils import hex_to_rgba, color_to_0_1_range


def _parse_length(value: str) -> Optional[float]:
    """
    Parse SVG length value (e.g., '12px', '10', '1.5em') to float.
    
    Args:
        value: SVG length string
        
    Returns:
        Float value or None if parsing fails
    """
    if not value:
        return None
    
    # Remove common units and parse the numeric value
    value = value.strip()
    # Match valid decimal numbers (with optional decimal point and digits)
    match = re.match(r'^(\d*\.?\d+)\s*(px|pt|em|rem|%)?$', value, re.IGNORECASE)
    if match:
        try:
            num = float(match.group(1))
            unit = match.group(2)
            # Convert pt to px (1pt = 1.333px approximately)
            if unit and unit.lower() == 'pt':
                num *= 1.333
            return num
        except ValueError:
            return None
    return None


def parse_svg(svg_file: str) -> Tuple[List[float], List[Tuple[str, str, 
                                                                  Dict[str, Any]]]]:
    """
    Parse an SVG file and extract relevant information.
    
    Args:
        svg_file: Path to the SVG file
        
    Returns:
        Tuple containing (svg_dimensions, path_data)
            - svg_dimensions: [width, height]
            - path_data: List of tuples (path_string, element_id, attributes_dict)
              where attributes_dict contains:
                - 'fill': fill color in 0-1 range [r, g, b, a]
                - 'stroke': stroke color in 0-1 range [r, g, b, a] or None
                - 'stroke_width': stroke width or None
    """
    try:
        doc = minidom.parse(svg_file)
    except Exception as e:
        raise ValueError(f"Failed to parse SVG file '{svg_file}': {e}")

    # Extract viewBox
    svg_element = doc.getElementsByTagName("svg")[0]
    viewbox_string = svg_element.getAttribute("viewBox")
    
    # Parse viewBox dimensions
    if "," in viewbox_string:
        sw_size = list(map(float, viewbox_string.split(",")[2:]))
    else:
        sw_size = list(map(float, viewbox_string.split(" ")[2:]))

    # Extract path data
    path_count = 0
    path_strings = []
    for path in doc.getElementsByTagName("path"):
        id_ = path.getAttribute("id") or f"path_{path_count}"
        d = path.getAttribute("d")
        
        # Parse fill attribute
        fill_attr = path.getAttribute("fill")
        if fill_attr and fill_attr.lower() != 'none':
            try:
                fill_rgba = hex_to_rgba(fill_attr)
                fill_color = color_to_0_1_range(fill_rgba)
            except (ValueError, AttributeError):
                fill_color = [1, 1, 1, 0]  # Transparent white
        else:
            fill_color = [1, 1, 1, 0]  # Transparent (no fill)
        
        # Parse stroke attribute
        stroke_attr = path.getAttribute("stroke")
        stroke_color = None
        if stroke_attr and stroke_attr.lower() != 'none':
            try:
                stroke_rgba = hex_to_rgba(stroke_attr)
                stroke_color = color_to_0_1_range(stroke_rgba)
            except (ValueError, AttributeError):
                stroke_color = None
        
        # Parse stroke-width attribute
        stroke_width_attr = path.getAttribute("stroke-width")
        stroke_width = None
        if stroke_width_attr:
            try:
                stroke_width = float(stroke_width_attr)
            except ValueError:
                stroke_width = None
        
        # Create attributes dictionary
        attrs = {
            'fill': fill_color,
            'stroke': stroke_color,
            'stroke_width': stroke_width
        }
        
        path_strings.append((d, id_, attrs))
        path_count += 1
    
    doc.unlink()
    return sw_size, path_strings


def parse_text_elements(svg_file: str) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Parse text elements from an SVG file.
    
    Args:
        svg_file: Path to the SVG file
        
    Returns:
        Tuple containing (svg_dimensions, text_elements)
            - svg_dimensions: [width, height]
            - text_elements: List of text element dictionaries containing:
                - 'text': The text content
                - 'x': X position
                - 'y': Y position
                - 'font_family': Font family name
                - 'font_size': Font size in pixels
                - 'font_weight': Font weight (normal, bold, 100-900)
                - 'font_style': Font style (normal, italic, oblique)
                - 'fill': Fill color in 0-1 range [r, g, b, a]
                - 'stroke': Stroke color in 0-1 range [r, g, b, a] or None
                - 'stroke_width': Stroke width or None
                - 'text_anchor': Text anchor (start, middle, end)
                - 'dominant_baseline': Baseline alignment
                - 'letter_spacing': Letter spacing
                - 'text_decoration': Text decoration (underline, line-through, etc.)
                - 'opacity': Opacity value (0-1)
                - 'id': Element ID
    """
    try:
        doc = minidom.parse(svg_file)
    except Exception as e:
        raise ValueError(f"Failed to parse SVG file '{svg_file}': {e}")

    # Extract viewBox
    svg_element = doc.getElementsByTagName("svg")[0]
    viewbox_string = svg_element.getAttribute("viewBox")
    
    # Parse viewBox dimensions
    if "," in viewbox_string:
        sw_size = list(map(float, viewbox_string.split(",")[2:]))
    else:
        sw_size = list(map(float, viewbox_string.split(" ")[2:]))

    text_elements = []
    text_count = 0
    
    for text_elem in doc.getElementsByTagName("text"):
        text_data = _parse_text_element(text_elem, text_count)
        if text_data:
            text_elements.append(text_data)
            text_count += 1
    
    doc.unlink()
    return sw_size, text_elements


def _parse_text_element(text_elem, text_count: int) -> Optional[Dict[str, Any]]:
    """
    Parse a single text element and extract its properties.
    
    Args:
        text_elem: DOM text element
        text_count: Current text element count for ID generation
        
    Returns:
        Dictionary of text properties or None if parsing fails
    """
    # Get text content (handle nested tspan elements)
    text_content = _get_text_content(text_elem)
    if not text_content:
        return None
    
    # Get element ID
    id_ = text_elem.getAttribute("id") or f"text_{text_count}"
    
    # Parse position
    x = _parse_length(text_elem.getAttribute("x")) or 0
    y = _parse_length(text_elem.getAttribute("y")) or 0
    
    # Parse font properties
    font_family = text_elem.getAttribute("font-family") or "sans-serif"
    font_size = _parse_length(text_elem.getAttribute("font-size")) or 16
    font_weight = text_elem.getAttribute("font-weight") or "normal"
    font_style = text_elem.getAttribute("font-style") or "normal"
    
    # Parse fill color
    fill_attr = text_elem.getAttribute("fill")
    if fill_attr and fill_attr.lower() != 'none':
        try:
            fill_rgba = hex_to_rgba(fill_attr)
            fill_color = color_to_0_1_range(fill_rgba)
        except (ValueError, AttributeError):
            fill_color = [0, 0, 0, 1]  # Default: opaque black
    else:
        fill_color = [0, 0, 0, 1]  # Default: opaque black
    
    # Parse stroke color
    stroke_attr = text_elem.getAttribute("stroke")
    stroke_color = None
    if stroke_attr and stroke_attr.lower() != 'none':
        try:
            stroke_rgba = hex_to_rgba(stroke_attr)
            stroke_color = color_to_0_1_range(stroke_rgba)
        except (ValueError, AttributeError):
            stroke_color = None
    
    # Parse stroke width
    stroke_width_attr = text_elem.getAttribute("stroke-width")
    stroke_width = None
    if stroke_width_attr:
        try:
            stroke_width = float(stroke_width_attr)
        except ValueError:
            stroke_width = None
    
    # Parse text alignment
    text_anchor = text_elem.getAttribute("text-anchor") or "start"
    dominant_baseline = text_elem.getAttribute("dominant-baseline") or "auto"
    
    # Parse letter spacing
    letter_spacing_attr = text_elem.getAttribute("letter-spacing")
    letter_spacing = _parse_length(letter_spacing_attr) if letter_spacing_attr else 0
    
    # Parse text decoration
    text_decoration = text_elem.getAttribute("text-decoration") or "none"
    
    # Parse opacity
    opacity_attr = text_elem.getAttribute("opacity")
    try:
        opacity = float(opacity_attr) if opacity_attr else 1.0
    except ValueError:
        opacity = 1.0
    
    return {
        'text': text_content,
        'x': x,
        'y': y,
        'font_family': font_family,
        'font_size': font_size,
        'font_weight': font_weight,
        'font_style': font_style,
        'fill': fill_color,
        'stroke': stroke_color,
        'stroke_width': stroke_width,
        'text_anchor': text_anchor,
        'dominant_baseline': dominant_baseline,
        'letter_spacing': letter_spacing,
        'text_decoration': text_decoration,
        'opacity': opacity,
        'id': id_
    }


def _get_text_content(text_elem) -> str:
    """
    Extract text content from a text element, handling nested tspan elements.
    
    Args:
        text_elem: DOM text element
        
    Returns:
        Combined text content string
    """
    text_parts = []
    
    for child in text_elem.childNodes:
        if child.nodeType == child.TEXT_NODE:
            text = child.data.strip()
            if text:
                text_parts.append(text)
        elif child.nodeName == 'tspan':
            # Recursively get text from tspan
            tspan_text = _get_text_content(child)
            if tspan_text:
                text_parts.append(tspan_text)
    
    return ' '.join(text_parts)
