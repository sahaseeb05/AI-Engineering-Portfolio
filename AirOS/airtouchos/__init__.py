"""AirTouchOS - touchless hand-gesture OS controller."""

from .config import AppConfig
from .gesture_engine import Gesture, GestureEngine, GestureOutput, HandMetrics
from .hand_tracker import HandObservation, HandTracker
from .os_controller import OSController
from .visual_overlay import VisualOverlay

__all__ = [
    "AppConfig",
    "Gesture",
    "GestureEngine",
    "GestureOutput",
    "HandMetrics",
    "HandObservation",
    "HandTracker",
    "OSController",
    "VisualOverlay",
]
__version__ = "1.0.0"
