"""
Central configuration for AegisGaze.

Every tunable threshold lives here so that clinicians / caregivers can adapt
the controller to a specific user without touching the processing code.
Values can be changed at runtime from the dashboard (the GUI mutates the
shared ``Settings`` instance) and are persisted to ``settings.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# File locations
# --------------------------------------------------------------------------- #
APP_DIR = Path.home() / ".aegisgaze"
SETTINGS_FILE = APP_DIR / "settings.json"
CALIBRATION_FILE = APP_DIR / "calibration.json"
MODEL_FILE = APP_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# --------------------------------------------------------------------------- #
# MediaPipe Face Mesh landmark indices (478-point topology, refine_landmarks)
# "Right" / "left" refer to the SUBJECT's eyes, not the image side.
# --------------------------------------------------------------------------- #
# Iris rings: centre point followed by 4 boundary points.
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]

# Eye corners (outer, inner) – both pairs run left->right in the raw image.
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)

# 6-point Eye Aspect Ratio sets: p1..p6 as in Soukupová & Čech (2016).
#   p1/p4 = horizontal corners, (p2,p6) and (p3,p5) = vertical lid pairs.
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Eye contours used only for drawing in the preview.
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                    466, 388, 387, 386, 385, 384, 398]

# Head-pitch reference points.
FOREHEAD = 10
NOSE_TIP = 1
CHIN = 152


@dataclass
class Settings:
    """User-tunable parameters (persisted as JSON)."""

    # ---- Camera ---------------------------------------------------------- #
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror_preview: bool = True

    # ---- Smoothing -------------------------------------------------------- #
    smoothing_method: str = "kalman"      # "kalman" or "ema"
    # 0 = raw/responsive ... 1 = very smooth/laggy. Drives both filters.
    smoothing_strength: float = 0.6
    cursor_deadzone_px: float = 4.0       # ignore micro-movements below this

    # ---- Dwell (left) click --------------------------------------------- #
    dwell_enabled: bool = True
    dwell_radius_px: float = 30.0
    dwell_time_s: float = 1.2
    dwell_cooldown_s: float = 1.0         # pause after a click before re-arming

    # ---- Blink (right) click -------------------------------------------- #
    blink_enabled: bool = True
    ear_closed_threshold: float = 0.19    # EAR below this = eye closed
    blink_hold_s: float = 0.8
    # If True, the LEFT eye must stay open while the right eye is closed
    # (a true wink). Many users with ALS cannot wink, so default is False;
    # the 0.8 s hold already rejects natural blinks (~0.1–0.4 s).
    blink_require_wink: bool = False
    blink_cooldown_s: float = 1.0

    # ---- Head-pitch scrolling ------------------------------------------- #
    scroll_enabled: bool = True
    pitch_deadzone: float = 0.045         # neutral band (fraction of face height)
    scroll_speed: float = 1.0             # wheel notches per step (grows with tilt)
    scroll_interval_s: float = 0.12       # min time between scroll events
    scroll_invert: bool = False

    # ---- Calibration ---------------------------------------------------- #
    calib_target_margin_px: int = 60      # distance of targets from screen edge
    calib_settle_s: float = 1.2           # time to move eyes onto target
    calib_sample_s: float = 1.5           # time spent collecting samples

    # ---- Hotkeys -------------------------------------------------------- #
    panic_key: str = "esc"                # always DISABLES control
    toggle_key: str = "f8"                # toggles control on/off

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls) -> "Settings":
        """Load settings from disk, falling back to defaults on any error."""
        settings = cls()
        try:
            if SETTINGS_FILE.exists():
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                valid = {f.name for f in fields(cls)}
                for key, value in data.items():
                    if key in valid:
                        setattr(settings, key, type(getattr(settings, key))(value))
        except Exception as exc:  # corrupt file, wrong types, ...
            log.warning("Could not load settings (%s); using defaults.", exc)
        return settings

    def save(self) -> None:
        try:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2),
                                     encoding="utf-8")
        except OSError as exc:
            log.error("Could not save settings: %s", exc)
