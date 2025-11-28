"""
Video export functionality.
"""
import cv2
import numpy as np
from typing import Optional, List


class VideoWriter:
    """
    Video writer for exporting animations.
    Supports MP4, AVI, and other formats.
    """
    
    def __init__(self, path: str, width: int, height: int, fps: int = 30,
                 codec: str = 'mp4v'):
        """
        Initialize the video writer.
        
        Args:
            path: Output file path
            width: Video width
            height: Video height
            fps: Frames per second
            codec: Video codec (default: mp4v for MP4 files)
        """
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer: Optional[cv2.VideoWriter] = None
        self._is_open = False
    
    def open(self) -> bool:
        """
        Open the video writer.
        
        Returns:
            True if opened successfully
        """
        try:
            self.writer = cv2.VideoWriter(
                self.path, 
                self.fourcc, 
                self.fps, 
                (self.width, self.height)
            )
            self._is_open = self.writer.isOpened()
            return self._is_open
        except Exception:
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the video.
        
        Args:
            frame: Image array (RGBA or BGR format)
            
        Returns:
            True if write was successful
        """
        if not self._is_open:
            if not self.open():
                return False
        
        try:
            # Convert RGBA to BGR if needed
            if frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                bgr = frame
            
            self.writer.write(bgr)
            return True
        except Exception:
            return False
    
    def close(self):
        """Close the video writer and finalize the file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._is_open = False
    
    @property
    def is_open(self) -> bool:
        """Check if the writer is open."""
        return self._is_open
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def write_video(frames: List[np.ndarray], path: str, fps: int = 30,
                codec: str = 'mp4v') -> bool:
    """
    Write a list of frames to a video file.
    
    Args:
        frames: List of image arrays
        path: Output file path
        fps: Frames per second
        codec: Video codec
        
    Returns:
        True if successful
    """
    if not frames:
        return False
    
    height, width = frames[0].shape[:2]
    
    with VideoWriter(path, width, height, fps, codec) as writer:
        for frame in frames:
            if not writer.write_frame(frame):
                return False
    
    return True
