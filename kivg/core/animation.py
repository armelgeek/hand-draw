"""
Animation module for OpenCV-based rendering.
Provides a standalone animation system that doesn't require Kivy.
"""

import time
from typing import Dict, Any, Callable, Optional, List
from .easing import AnimationTransition


class Animation:
    """
    Simple animation class for property interpolation.
    Replaces kivy.animation.Animation for headless rendering.
    """
    
    # Class-level tracking of all active animations
    _instances: set = set()
    
    def __init__(self, **kw):
        """
        Initialize animation with target property values.
        
        Args:
            d or duration: Animation duration in seconds (default: 1.0)
            t or transition: Transition function name (default: 'linear')
            **kw: Target property values to animate to
        """
        self._duration = kw.pop('d', kw.pop('duration', 1.0))
        self._transition_name = kw.pop('t', kw.pop('transition', 'linear'))
        
        if isinstance(self._transition_name, str):
            self._transition = getattr(AnimationTransition, self._transition_name)
        else:
            self._transition = self._transition_name
            
        self._animated_properties = kw
        self._widgets: Dict[int, dict] = {}
        
        # Callbacks
        self._on_start_callbacks: List[Callable] = []
        self._on_progress_callbacks: List[Callable] = []
        self._on_complete_callbacks: List[Callable] = []
    
    @property
    def duration(self) -> float:
        """Return the duration of the animation."""
        return self._duration
    
    @property
    def transition(self):
        """Return the transition function of the animation."""
        return self._transition
    
    @property
    def animated_properties(self) -> dict:
        """Return the properties being animated."""
        return self._animated_properties
    
    def bind(self, on_start: Optional[Callable] = None,
             on_progress: Optional[Callable] = None,
             on_complete: Optional[Callable] = None):
        """
        Bind callbacks to animation events.
        
        Args:
            on_start: Called when animation starts
            on_progress: Called on each animation step (receives widget, progress)
            on_complete: Called when animation completes
        """
        if on_start:
            self._on_start_callbacks.append(on_start)
        if on_progress:
            self._on_progress_callbacks.append(on_progress)
        if on_complete:
            self._on_complete_callbacks.append(on_complete)
    
    def unbind(self, on_start: Optional[Callable] = None,
               on_progress: Optional[Callable] = None,
               on_complete: Optional[Callable] = None):
        """Unbind callbacks from animation events."""
        if on_start and on_start in self._on_start_callbacks:
            self._on_start_callbacks.remove(on_start)
        if on_progress and on_progress in self._on_progress_callbacks:
            self._on_progress_callbacks.remove(on_progress)
        if on_complete and on_complete in self._on_complete_callbacks:
            self._on_complete_callbacks.remove(on_complete)
    
    def _dispatch(self, event: str, widget: Any, progress: Optional[float] = None):
        """Dispatch an event to registered callbacks."""
        if event == 'on_start':
            for callback in self._on_start_callbacks:
                callback(self, widget)
        elif event == 'on_progress':
            for callback in self._on_progress_callbacks:
                callback(self, widget, progress)
        elif event == 'on_complete':
            for callback in self._on_complete_callbacks:
                callback(self, widget)
    
    def start(self, widget: Any):
        """
        Start the animation on a widget/object.
        
        Args:
            widget: Object with properties to animate
        """
        self.stop(widget)
        self._initialize(widget)
        Animation._instances.add(self)
        self._dispatch('on_start', widget)
    
    def stop(self, widget: Any):
        """Stop the animation, triggering on_complete."""
        uid = id(widget)
        if uid in self._widgets:
            del self._widgets[uid]
            self._dispatch('on_complete', widget)
        self.cancel(widget)
    
    def cancel(self, widget: Any):
        """Cancel the animation without triggering on_complete."""
        uid = id(widget)
        if uid in self._widgets:
            del self._widgets[uid]
        if not self._widgets and self in Animation._instances:
            Animation._instances.remove(self)
    
    @staticmethod
    def cancel_all(widget: Any, *largs):
        """Cancel all animations on a widget."""
        for animation in list(Animation._instances):
            if largs:
                for prop in largs:
                    animation.cancel_property(widget, prop)
            else:
                animation.cancel(widget)
    
    def cancel_property(self, widget: Any, prop: str):
        """Cancel animation of a specific property."""
        uid = id(widget)
        if uid in self._widgets:
            self._widgets[uid]['properties'].pop(prop, None)
            if not self._widgets[uid]['properties']:
                self.cancel(widget)
    
    def have_properties_to_animate(self, widget: Any) -> bool:
        """Check if widget still has properties to animate."""
        uid = id(widget)
        return uid in self._widgets and bool(self._widgets[uid]['properties'])
    
    def _initialize(self, widget: Any):
        """Initialize animation state for widget."""
        uid = id(widget)
        self._widgets[uid] = {
            'widget': widget,
            'properties': {},
            'start_time': None
        }
        
        # Store initial values
        props = self._widgets[uid]['properties']
        for key, target in self._animated_properties.items():
            original = getattr(widget, key)
            if isinstance(original, (tuple, list)):
                original = list(original)
            elif isinstance(original, dict):
                original = original.copy()
            props[key] = (original, target)
    
    def _update(self, dt: float, widget: Any) -> bool:
        """
        Update animation state.
        
        Args:
            dt: Time delta since last update
            widget: Widget being animated
            
        Returns:
            True if animation is still running, False if complete
        """
        uid = id(widget)
        if uid not in self._widgets:
            return False
        
        anim = self._widgets[uid]
        
        if anim['start_time'] is None:
            anim['start_time'] = time.time()
        
        elapsed = time.time() - anim['start_time']
        
        # Calculate progress
        if self._duration > 0:
            progress = min(1.0, elapsed / self._duration)
        else:
            progress = 1.0
        
        t = self._transition(progress)
        
        # Update properties
        for key, (start_val, end_val) in anim['properties'].items():
            value = self._calculate(start_val, end_val, t)
            setattr(widget, key, value)
        
        self._dispatch('on_progress', widget, progress)
        
        # Check if complete
        if progress >= 1.0:
            self.stop(widget)
            return False
        
        return True
    
    def _calculate(self, a, b, t: float):
        """Calculate interpolated value."""
        if isinstance(a, (list, tuple)):
            tp = type(a)
            return tp([self._calculate(a[i], b[i], t) for i in range(len(a))])
        elif isinstance(a, dict):
            return {k: self._calculate(a.get(k, 0), b.get(k, 0), t) for k in a}
        else:
            return a * (1 - t) + b * t
    
    def animate_sync(self, widget: Any, fps: int = 60, 
                     on_frame: Optional[Callable] = None) -> List[Any]:
        """
        Run animation synchronously and return all frames.
        
        This is useful for generating animation frames for export.
        
        Args:
            widget: Object with properties to animate
            fps: Frames per second
            on_frame: Optional callback called for each frame
            
        Returns:
            List of frame states (copies of widget state at each frame)
        """
        frames = []
        num_frames = max(1, int(self._duration * fps))
        
        # Store initial values
        initial_values = {}
        for key in self._animated_properties:
            initial_values[key] = getattr(widget, key)
        
        for i in range(num_frames + 1):
            progress = i / num_frames if num_frames > 0 else 1.0
            t = self._transition(progress)
            
            # Update properties
            for key, target in self._animated_properties.items():
                start_val = initial_values[key]
                value = self._calculate(start_val, target, t)
                setattr(widget, key, value)
            
            # Call frame callback
            if on_frame:
                on_frame(widget, progress)
            
            frames.append(progress)
        
        return frames
    
    def __add__(self, other: 'Animation') -> 'Sequence':
        """Create sequential animation."""
        return Sequence(self, other)
    
    def __and__(self, other: 'Animation') -> 'Parallel':
        """Create parallel animation."""
        return Parallel(self, other)


class CompoundAnimation(Animation):
    """Base class for compound animations (Sequence and Parallel)."""
    
    def __init__(self):
        super().__init__()
        self.anim1: Optional[Animation] = None
        self.anim2: Optional[Animation] = None
    
    def have_properties_to_animate(self, widget: Any) -> bool:
        return (self.anim1.have_properties_to_animate(widget) or
                self.anim2.have_properties_to_animate(widget))
    
    @property
    def animated_properties(self) -> dict:
        props = {}
        props.update(self.anim1.animated_properties)
        props.update(self.anim2.animated_properties)
        return props


class Sequence(CompoundAnimation):
    """Sequential animation - runs animations one after another."""
    
    def __init__(self, anim1: Animation, anim2: Animation):
        super().__init__()
        self.anim1 = anim1
        self.anim2 = anim2
        self.repeat = False
        self._current_anim = None
        self._widgets: Dict[int, bool] = {}
    
    @property
    def duration(self) -> float:
        return self.anim1.duration + self.anim2.duration
    
    def start(self, widget: Any):
        """Start the sequential animation."""
        self.stop(widget)
        uid = id(widget)
        self._widgets[uid] = True
        Animation._instances.add(self)
        self._dispatch('on_start', widget)
        
        # Set up completion handler for first animation
        def on_anim1_complete(anim, w):
            if id(w) in self._widgets:
                self.anim2.start(w)
        
        def on_anim2_complete(anim, w):
            if id(w) not in self._widgets:
                return
            if self.repeat:
                self.anim1.start(w)
            else:
                self._dispatch('on_complete', w)
                self.cancel(w)
        
        self.anim1.bind(on_complete=on_anim1_complete)
        self.anim2.bind(on_complete=on_anim2_complete)
        
        self._current_anim = self.anim1
        self.anim1.start(widget)
    
    def stop(self, widget: Any):
        """Stop the animation."""
        uid = id(widget)
        if uid in self._widgets:
            del self._widgets[uid]
            self.anim1.stop(widget)
            self.anim2.stop(widget)
            self._dispatch('on_complete', widget)
    
    def cancel(self, widget: Any):
        """Cancel the animation."""
        uid = id(widget)
        if uid in self._widgets:
            del self._widgets[uid]
        self.anim1.cancel(widget)
        self.anim2.cancel(widget)
        if self in Animation._instances:
            Animation._instances.remove(self)


class Parallel(CompoundAnimation):
    """Parallel animation - runs animations simultaneously."""
    
    def __init__(self, anim1: Animation, anim2: Animation):
        super().__init__()
        self.anim1 = anim1
        self.anim2 = anim2
        self._widgets: Dict[int, dict] = {}
    
    @property
    def duration(self) -> float:
        return max(self.anim1.duration, self.anim2.duration)
    
    def start(self, widget: Any):
        """Start both animations in parallel."""
        self.stop(widget)
        uid = id(widget)
        self._widgets[uid] = {'complete': 0}
        Animation._instances.add(self)
        self._dispatch('on_start', widget)
        
        def on_anim_complete(anim, w):
            uid = id(w)
            if uid in self._widgets:
                self._widgets[uid]['complete'] += 1
                if self._widgets[uid]['complete'] >= 2:
                    self.stop(w)
        
        self.anim1.bind(on_complete=on_anim_complete)
        self.anim2.bind(on_complete=on_anim_complete)
        
        self.anim1.start(widget)
        self.anim2.start(widget)
    
    def stop(self, widget: Any):
        """Stop both animations."""
        uid = id(widget)
        if uid in self._widgets:
            del self._widgets[uid]
            self._dispatch('on_complete', widget)
        self.anim1.cancel(widget)
        self.anim2.cancel(widget)
        if self in Animation._instances:
            Animation._instances.remove(self)
    
    def cancel(self, widget: Any):
        """Cancel both animations."""
        uid = id(widget)
        if uid in self._widgets:
            del self._widgets[uid]
        self.anim1.cancel(widget)
        self.anim2.cancel(widget)
        if self in Animation._instances:
            Animation._instances.remove(self)
