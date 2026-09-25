"""
GestureEngine: landmarks -> gesture state machine -> high-level intents.

The engine is deliberately free of any OS or UI side-effects.  It consumes a
``HandObservation`` (or ``None`` when no hand is visible) plus a timestamp and
returns a ``GestureOutput`` describing *what should happen* (move the cursor,
click, start a drag, close a window, set the volume, ...).  The
``OSController`` then carries those intents out.  Keeping the engine pure
makes it deterministic and unit-testable with synthetic landmarks.

Gesture priority (highest first) on every frame with a visible hand:

    1. Open-palm hold  -> toggles PAUSE (the only gesture active while paused)
    2. Fist            -> Alt+F4 (confirm time + 1.5 s debounce + latch)
    3. Volume pose     -> thumb, index & pinky out; middle & ring folded
    4. Right pinch     -> middle tip (12) + thumb tip (4), fires on contact
    5. Left pinch      -> index tip (8) + thumb tip (4):
                          released quickly = click, held = drag & drop
    6. Pointer         -> index tip (8) drives the cursor
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .config import AppConfig
from .hand_tracker import (
    HandObservation,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
)

# Finger order used for every per-finger array in this module.
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
_PIPS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
_PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class Gesture(str, Enum):
    """Primary gesture state for the current frame (used for display)."""

    NO_HAND = "NO HAND"
    POINTER = "POINTER"
    PINCH = "PINCH"              # left pinch held, not yet a drag
    DRAG = "DRAG"
    RIGHT_PINCH = "RIGHT CLICK"
    FIST = "FIST"
    VOLUME = "VOLUME"
    OPEN_PALM = "OPEN PALM"
    PAUSED = "PAUSED"


@dataclass(frozen=True)
class HandMetrics:
    """Scale-normalised geometric features of one hand.

    All ``*_ratio``/distance values are divided by the palm length
    (wrist -> middle MCP), so they do not depend on distance to the camera.
    """

    scale_px: float                         # palm length in pixels
    extended: Tuple[bool, bool, bool, bool, bool]
    tip_to_wrist: Tuple[float, float, float, float, float]
    left_pinch: float                       # |index tip - thumb tip|
    right_pinch: float                      # |middle tip - thumb tip|
    volume_span: float                      # |index tip.y - thumb tip.y|
    palm_center: Tuple[float, float]        # palm centroid / max(frame w, h)
    pointer: Tuple[float, float]            # index tip, normalised [0, 1]


@dataclass
class GestureOutput:
    """Intents produced for one frame.  Event flags are one-shot (edge)."""

    gesture: Gesture
    metrics: Optional[HandMetrics] = None
    cursor_target: Optional[Tuple[float, float]] = None  # None = keep cursor still
    precision: bool = False          # a pinch is forming: slow the cursor down
    left_click: bool = False
    right_click: bool = False
    drag_start: bool = False
    drag_end: bool = False
    close_window: bool = False
    volume_level: Optional[float] = None   # 0..1 target while volume mode is active
    paused: bool = False
    pause_toggled: bool = False
    palm_progress: float = 0.0       # 0..1 progress of the pause hold
    fist_progress: float = 0.0       # 0..1 progress of the fist confirmation
    left_pinch_active: bool = False
    right_pinch_active: bool = False
    events: list = field(default_factory=list)  # human-readable event labels


def compute_metrics(obs: HandObservation, cfg: AppConfig) -> HandMetrics:
    """Derive rotation- and scale-robust features from raw landmarks."""
    px = obs.landmarks_px

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(px[a] - px[b]))

    scale = max(dist(WRIST, MIDDLE_MCP), 1e-6)

    # Fingers (not thumb): tip farther from the wrist than the PIP joint.
    extended = [False] * 5
    for i in range(1, 5):
        extended[i] = dist(_TIPS[i], WRIST) > dist(_PIPS[i], WRIST) * cfg.finger_extended_ratio
    # Thumb: moves sideways, so compare against the pinky MCP instead of the wrist.
    extended[0] = dist(THUMB_TIP, PINKY_MCP) > dist(THUMB_IP, PINKY_MCP) * cfg.thumb_extended_ratio

    tip_to_wrist = tuple(dist(t, WRIST) / scale for t in _TIPS)
    width, height = obs.frame_size
    palm = px[list(_PALM_POINTS)].mean(axis=0) / float(max(width, height))

    return HandMetrics(
        scale_px=scale,
        extended=tuple(extended),  # type: ignore[arg-type]
        tip_to_wrist=tip_to_wrist,  # type: ignore[arg-type]
        left_pinch=dist(INDEX_TIP, THUMB_TIP) / scale,
        right_pinch=dist(MIDDLE_TIP, THUMB_TIP) / scale,
        volume_span=abs(float(px[INDEX_TIP, 1] - px[THUMB_TIP, 1])) / scale,
        palm_center=(float(palm[0]), float(palm[1])),
        pointer=(float(obs.landmarks_norm[INDEX_TIP, 0]), float(obs.landmarks_norm[INDEX_TIP, 1])),
    )


class GestureEngine:
    """Stateful, time-aware gesture recogniser.  Call :meth:`update` per frame."""

    _IDLE, _PRESSED, _DRAGGING = "idle", "pressed", "dragging"

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._paused = False
        self._last_seen: Optional[float] = None
        # Left pinch / drag state machine.
        self._left_state = self._IDLE
        self._left_since = 0.0
        # Right pinch latch.
        self._right_latched = False
        self._last_right_click = -math.inf
        # Fist.
        self._fist_since: Optional[float] = None
        self._fist_fired = False
        self._last_close = -math.inf
        # Volume.
        self._volume_since: Optional[float] = None
        self._volume_active = False
        self._volume_level: Optional[float] = None
        # Open-palm pause hold.
        self._palm_since: Optional[float] = None
        self._palm_anchor: Optional[Tuple[float, float]] = None
        self._palm_latched = False

    # ----------------------------------------------------------------- public
    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def dragging(self) -> bool:
        return self._left_state == self._DRAGGING

    def set_paused(self, paused: bool) -> None:
        """Pause/unpause from outside (e.g. keyboard).  Drops transient state;
        the caller is responsible for releasing any held mouse button."""
        self._paused = paused
        self._reset_transient()

    def update(self, obs: Optional[HandObservation], now: float) -> GestureOutput:
        """Advance the state machine by one frame and return the intents."""
        if obs is None:
            return self._on_no_hand(now)
        self._last_seen = now

        m = compute_metrics(obs, self._cfg)
        out = GestureOutput(gesture=Gesture.POINTER, metrics=m, paused=self._paused)
        is_palm = all(m.extended)

        # 1) Open-palm hold toggles pause (works in both states).
        out.palm_progress, toggled = self._update_palm_hold(is_palm, m.palm_center, now)
        if toggled:
            self._paused = not self._paused
            out.pause_toggled = True
            out.paused = self._paused
            out.events.append("TRACKING PAUSED" if self._paused else "TRACKING RESUMED")
            if self._paused:
                out.drag_end = self._left_state == self._DRAGGING
                self._reset_transient()
        if self._paused:
            out.gesture = Gesture.OPEN_PALM if out.palm_progress > 0 else Gesture.PAUSED
            return out

        # 2) Fist -> Alt+F4.
        if self._is_fist(m):
            return self._handle_fist(out, now)
        self._fist_since = None
        self._fist_fired = False

        # 3) Volume pose (never interrupts an active drag).
        if self._left_state != self._DRAGGING and self._is_volume_pose(m):
            return self._handle_volume(out, m, now)
        self._volume_since = None
        self._volume_active = False
        self._volume_level = None

        # 4) Right pinch -> right click on contact.
        if self._handle_right_pinch(out, m, now):
            return out

        # 5) Left pinch -> click on quick release, drag when held.
        if self._handle_left_pinch(out, m, now):
            return out

        # 6) Pointer.
        out.gesture = Gesture.OPEN_PALM if (is_palm and out.palm_progress > 0) else Gesture.POINTER
        out.cursor_target = m.pointer
        zone = self._cfg.precision_zone_factor
        out.precision = (
            m.left_pinch < self._cfg.left_pinch_on * zone
            or m.right_pinch < self._cfg.right_pinch_on * zone
        )
        return out

    # ---------------------------------------------------------------- no hand
    def _on_no_hand(self, now: float) -> GestureOutput:
        out = GestureOutput(gesture=Gesture.PAUSED if self._paused else Gesture.NO_HAND,
                            paused=self._paused)
        # Tolerate a few dropped frames before tearing state down, so a single
        # missed detection does not cancel a drag.
        if self._last_seen is not None and now - self._last_seen < self._cfg.lost_hand_release_s:
            return out
        if self._left_state == self._DRAGGING:
            out.drag_end = True
            out.events.append("DROP (hand lost)")
        self._reset_transient()
        self._palm_since = None
        self._palm_anchor = None
        self._palm_latched = False
        self._last_seen = None
        return out

    def _reset_transient(self) -> None:
        self._left_state = self._IDLE
        self._right_latched = False
        self._fist_since = None
        self._fist_fired = False
        self._volume_since = None
        self._volume_active = False
        self._volume_level = None

    # ------------------------------------------------------------------ palm
    def _update_palm_hold(self, is_palm: bool, center: Tuple[float, float], now: float
                          ) -> Tuple[float, bool]:
        """Return (progress 0..1, toggled?) for the stationary open-palm hold."""
        if not is_palm:
            self._palm_since = None
            self._palm_anchor = None
            self._palm_latched = False      # hand closed: next hold may toggle again
            return 0.0, False
        if self._palm_latched:
            return 0.0, False               # already toggled; wait for palm release
        moved = (
            self._palm_anchor is None
            or math.dist(center, self._palm_anchor) > self._cfg.palm_still_radius
        )
        if self._palm_since is None or moved:
            self._palm_since = now
            self._palm_anchor = center
            return 0.0, False
        progress = (now - self._palm_since) / self._cfg.palm_hold_s
        if progress >= 1.0:
            self._palm_latched = True
            self._palm_since = None
            return 1.0, True
        return progress, False

    # ------------------------------------------------------------------ fist
    def _is_fist(self, m: HandMetrics) -> bool:
        fingers_curled = all(r < self._cfg.fist_tip_ratio for r in m.tip_to_wrist[1:])
        thumb_tucked = m.tip_to_wrist[0] < self._cfg.fist_thumb_ratio
        return fingers_curled and thumb_tucked

    def _handle_fist(self, out: GestureOutput, now: float) -> GestureOutput:
        # Closing the hand can pass through a pinch: cancel it without clicking,
        # and never close a window while a mouse button is held down.
        if self._left_state == self._DRAGGING:
            out.drag_end = True
        self._left_state = self._IDLE
        self._right_latched = False
        self._volume_since = None
        self._volume_active = False
        self._volume_level = None

        if self._fist_since is None:
            self._fist_since = now
        out.gesture = Gesture.FIST
        out.fist_progress = min(1.0, (now - self._fist_since) / max(self._cfg.fist_confirm_s, 1e-6))
        if (
            out.fist_progress >= 1.0
            and not self._fist_fired
            and now - self._last_close >= self._cfg.fist_debounce_s
        ):
            out.close_window = True
            out.events.append("CLOSE WINDOW (Alt+F4)")
            self._fist_fired = True         # one trigger per fist
            self._last_close = now          # and at most one per debounce window
        return out

    # ---------------------------------------------------------------- volume
    def _is_volume_pose(self, m: HandMetrics) -> bool:
        thumb, index, middle, ring, pinky = m.extended
        base = pinky and not middle and not ring and m.right_pinch > self._cfg.right_pinch_off
        if self._volume_active or self._volume_since is not None:
            # Hysteresis: once engaged, thumb/index may bend while the span
            # shrinks, so only the anchor fingers are required to stay put.
            return base
        return base and thumb and index

    def _handle_volume(self, out: GestureOutput, m: HandMetrics, now: float) -> GestureOutput:
        # Entering the pose can briefly look like a pinch; cancel it silently.
        self._left_state = self._IDLE
        self._right_latched = False
        out.gesture = Gesture.VOLUME
        if self._volume_since is None:
            self._volume_since = now
        if not self._volume_active:
            if now - self._volume_since < self._cfg.volume_confirm_s:
                return out
            self._volume_active = True
            out.events.append("VOLUME MODE")

        raw = float(np.interp(m.volume_span,
                              [self._cfg.volume_span_min, self._cfg.volume_span_max],
                              [0.0, 1.0]))
        if self._volume_level is None:
            self._volume_level = raw
        else:
            a = self._cfg.volume_smoothing
            self._volume_level = a * raw + (1.0 - a) * self._volume_level
        step = max(self._cfg.volume_step_pct, 0.1) / 100.0
        out.volume_level = float(np.clip(round(self._volume_level / step) * step, 0.0, 1.0))
        return out

    # ---------------------------------------------------------- right pinch
    def _handle_right_pinch(self, out: GestureOutput, m: HandMetrics, now: float) -> bool:
        cfg = self._cfg
        if self._right_latched:
            if m.right_pinch > cfg.right_pinch_off:
                self._right_latched = False
                return False
            out.gesture = Gesture.RIGHT_PINCH
            out.right_pinch_active = True
            return True                     # hold cursor still until released
        if (
            self._left_state == self._IDLE
            and m.right_pinch < cfg.right_pinch_on
            and m.right_pinch < m.left_pinch    # the closer finger wins
            and now - self._last_right_click >= cfg.right_click_cooldown_s
        ):
            self._right_latched = True
            self._last_right_click = now
            out.gesture = Gesture.RIGHT_PINCH
            out.right_pinch_active = True
            out.right_click = True
            out.events.append("RIGHT CLICK")
            return True
        return False

    # ----------------------------------------------------------- left pinch
    def _handle_left_pinch(self, out: GestureOutput, m: HandMetrics, now: float) -> bool:
        cfg = self._cfg
        if self._left_state == self._IDLE:
            right_closer = m.right_pinch < cfg.right_pinch_on and m.right_pinch < m.left_pinch
            if m.left_pinch < cfg.left_pinch_on and not right_closer:
                self._left_state = self._PRESSED
                self._left_since = now

        if self._left_state == self._PRESSED:
            if m.left_pinch > cfg.left_pinch_off:
                # Released before the drag threshold: a plain click.  The
                # cursor was frozen while pinched, so the click lands exactly
                # where the pinch started.
                self._left_state = self._IDLE
                out.left_click = True
                out.events.append("LEFT CLICK")
                return False                # fall through: pointer resumes
            if now - self._left_since >= cfg.drag_hold_s:
                self._left_state = self._DRAGGING
                out.drag_start = True
                out.events.append("DRAG START")
            else:
                out.gesture = Gesture.PINCH
                out.left_pinch_active = True
                return True                 # cursor frozen while deciding

        if self._left_state == self._DRAGGING:
            if m.left_pinch > cfg.left_pinch_off:
                self._left_state = self._IDLE
                out.drag_end = True
                out.events.append("DROP")
                return False
            out.gesture = Gesture.DRAG
            out.left_pinch_active = True
            out.cursor_target = m.pointer
            return True
        return False
