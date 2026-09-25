"""
Central configuration for AirTouchOS.

All tunable thresholds live in one dataclass so they can be calibrated
without touching any logic.  Every distance threshold for the hand is
*scale-normalised*: it is expressed in multiples of the "palm length"
(distance from the Wrist, landmark 0, to the Middle-finger MCP, landmark 9).
That makes the thresholds independent of how far the hand is from the
camera.

A JSON file can override any subset of fields:

    python main.py --config my_thresholds.json

Use ``python main.py --dump-config my_thresholds.json`` to write the
defaults to disk as a starting point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class AppConfig:
    # ------------------------------------------------------------------ camera
    camera_index: int = 0                 # OpenCV device index of the webcam
    frame_width: int = 640                # Requested capture width (px)
    frame_height: int = 480               # Requested capture height (px)
    camera_fps: int = 30                  # Requested capture frame rate
    mirror: bool = True                   # Horizontal flip for natural feedback

    # ------------------------------------------------------- hand landmarker
    model_path: str = "models/hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    min_detection_confidence: float = 0.70
    min_presence_confidence: float = 0.60
    min_tracking_confidence: float = 0.60

    # ------------------------------------------------------- cursor mapping
    # Fraction of the camera frame ignored on each side.  The inner "active
    # region" is stretched to the full screen, so the fingertip reaches the
    # screen corners well before it leaves the camera's field of view.
    margin_x: float = 0.15
    margin_y: float = 0.18
    # Exponential moving average factors (0 < alpha <= 1).  Higher = snappier,
    # lower = smoother.  The effective alpha is interpolated between the min
    # and max based on how fast the fingertip is moving (adaptive EMA): slow,
    # precise motion is smoothed heavily, fast sweeps stay responsive.
    ema_alpha_min: float = 0.18
    ema_alpha_max: float = 0.55
    ema_speed_low_px: float = 4.0         # speed (px/frame) at/below -> alpha_min
    ema_speed_high_px: float = 60.0       # speed (px/frame) at/above -> alpha_max
    precision_alpha_factor: float = 0.35  # alpha multiplier while a pinch is forming
    cursor_deadzone_px: float = 1.5       # ignore sub-pixel noise below this movement

    # ----------------------------------------------------- finger pose tests
    # Finger counts as extended if dist(tip, wrist) > dist(pip, wrist) * ratio.
    finger_extended_ratio: float = 1.10
    # Thumb counts as extended if dist(tip, pinky_mcp) > dist(ip, pinky_mcp) * ratio.
    thumb_extended_ratio: float = 1.08

    # ------------------------------------------------------------ pinches
    # Hysteresis: engage below *_on, release above *_off (off > on).
    left_pinch_on: float = 0.26
    left_pinch_off: float = 0.38
    right_pinch_on: float = 0.26
    right_pinch_off: float = 0.38
    precision_zone_factor: float = 1.8    # slow the cursor when pinch < on * factor
    drag_hold_s: float = 0.35             # held pinch longer than this = drag
    right_click_cooldown_s: float = 0.40

    # ------------------------------------------------------------ fist
    fist_tip_ratio: float = 1.05          # every finger tip-to-wrist must be below this
    fist_thumb_ratio: float = 1.20        # thumb tip-to-wrist must be below this
    fist_confirm_s: float = 0.30          # fist must be held this long before firing
    fist_debounce_s: float = 1.50         # minimum gap between two Alt+F4 triggers

    # ------------------------------------------------------------ volume
    # Volume pose: thumb + index + pinky extended, middle + ring folded.
    # Vertical |index_tip.y - thumb_tip.y| (normalised) maps onto 0..100 %.
    volume_span_min: float = 0.15
    volume_span_max: float = 1.10
    volume_confirm_s: float = 0.30        # pose must be held before volume changes
    volume_smoothing: float = 0.30        # EMA alpha applied to the volume level
    volume_step_pct: float = 2.0          # quantisation of the volume level (percent)
    # Key-press fallback (when pycaw is unavailable): zones on the 0..1 level.
    volume_key_up_zone: float = 0.65
    volume_key_down_zone: float = 0.35
    volume_key_interval_s: float = 0.08

    # ------------------------------------------------------------ pause
    palm_hold_s: float = 2.0              # open palm held this long toggles pause
    palm_still_radius: float = 0.035      # max palm drift (fraction of frame) while holding

    # ------------------------------------------------------------ safety
    protect_desktop_from_close: bool = True   # never send Alt+F4 to desktop / taskbar
    lost_hand_release_s: float = 0.25         # release held mouse buttons after hand loss

    # ------------------------------------------------------------ UI
    window_name: str = "AirTouchOS"
    preview_scale: float = 1.0
    window_topmost: bool = True
    show_calibration: bool = False

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_json(cls, path: str | Path, base: "AppConfig | None" = None) -> "AppConfig":
        """Return a config with the fields found in *path* overriding *base*."""
        data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name: f for f in fields(cls)}
        unknown = sorted(set(data) - set(valid))
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {', '.join(unknown)}")
        # Coerce each value to the type of the default so "30" or 30.0 work.
        coerced: Dict[str, Any] = {}
        defaults = base or cls()
        for key, value in data.items():
            default_value = getattr(defaults, key)
            if isinstance(default_value, bool):
                coerced[key] = bool(value)
            elif isinstance(default_value, int):
                coerced[key] = int(value)
            elif isinstance(default_value, float):
                coerced[key] = float(value)
            else:
                coerced[key] = value
        return replace(defaults, **coerced).validated()

    def to_json(self, path: str | Path) -> None:
        """Write every field to *path* as pretty-printed JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def validated(self) -> "AppConfig":
        """Raise ``ValueError`` if thresholds are inconsistent; return self."""
        problems = []
        if not 0.0 <= self.margin_x < 0.5 or not 0.0 <= self.margin_y < 0.5:
            problems.append("margin_x / margin_y must be in [0, 0.5)")
        if not 0.0 < self.ema_alpha_min <= self.ema_alpha_max <= 1.0:
            problems.append("need 0 < ema_alpha_min <= ema_alpha_max <= 1")
        if self.ema_speed_high_px <= self.ema_speed_low_px:
            problems.append("ema_speed_high_px must exceed ema_speed_low_px")
        if self.left_pinch_off <= self.left_pinch_on:
            problems.append("left_pinch_off must exceed left_pinch_on (hysteresis)")
        if self.right_pinch_off <= self.right_pinch_on:
            problems.append("right_pinch_off must exceed right_pinch_on (hysteresis)")
        if self.volume_span_max <= self.volume_span_min:
            problems.append("volume_span_max must exceed volume_span_min")
        if self.volume_key_down_zone >= self.volume_key_up_zone:
            problems.append("volume_key_down_zone must be below volume_key_up_zone")
        if self.fist_debounce_s < 0 or self.palm_hold_s <= 0:
            problems.append("fist_debounce_s must be >= 0 and palm_hold_s > 0")
        if problems:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))
        return self
