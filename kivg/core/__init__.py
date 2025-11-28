"""
Core components for OpenCV-based SVG rendering.
"""
from .canvas import OpenCVCanvas
from .animation import Animation
from .easing import AnimationTransition

__all__ = ["OpenCVCanvas", "Animation", "AnimationTransition"]
