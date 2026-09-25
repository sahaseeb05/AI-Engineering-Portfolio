"""
Gesture recognisers. Each is a small, pure state machine that consumes one
measurement per frame plus a timestamp and reports whether its action fired.
They contain no I/O, which keeps them unit-testable.

* ``DwellClicker``  – left click when the cursor rests inside a radius.
* ``BlinkDetector`` – right click on a sustained right-eye closure.
* ``HeadScroller``  – scroll proportional to head pitch beyond a dead-zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
class DwellClicker:
    """
    Fires once the cursor has stayed within ``radius`` px of an anchor point
    for ``dwell_s`` seconds. After firing it stays disarmed until the cursor
    leaves the circle (prevents machine-gun clicking while the user reads)
    and until the cooldown has elapsed.
    """

    def __init__(self, radius: float, dwell_s: float, cooldown_s: float):
        self.radius = radius
        self.dwell_s = dwell_s
        self.cooldown_s = cooldown_s
        self.reset()

    def reset(self) -> None:
        self._anchor: Optional[Tuple[float, float]] = None
        self._start = 0.0
        self._armed = True
        self._cooldown_until = 0.0

    def update(self, pos: Tuple[float, float], now: float) -> Tuple[float, bool]:
        """Returns (progress 0..1, clicked_now)."""
        if self._anchor is None or math.dist(pos, self._anchor) > self.radius:
            # Cursor moved away: start a new dwell at the current position.
            self._anchor = pos
            self._start = now
            self._armed = True
            return 0.0, False

        if not self._armed:
            return 0.0, False
        if now < self._cooldown_until:
            self._start = now  # dwell timer only starts once cooldown ends
            return 0.0, False

        progress = (now - self._start) / self.dwell_s
        if progress >= 1.0:
            self._armed = False
            self._cooldown_until = now + self.cooldown_s
            return 1.0, True
        return progress, False

    @property
    def anchor(self) -> Optional[Tuple[float, float]]:
        return self._anchor


# --------------------------------------------------------------------------- #
@dataclass
class BlinkState:
    right_closed: bool = False
    left_closed: bool = False
    hold_progress: float = 0.0   # right-eye hold towards the right-click
    fired: bool = False


class BlinkDetector:
    """
    Right click when the RIGHT eye's EAR stays below threshold for
    ``hold_s`` seconds. Fires once per closure. Optionally requires the left
    eye to remain open (a deliberate wink).
    """

    def __init__(self, threshold: float, hold_s: float, require_wink: bool,
                 cooldown_s: float):
        self.threshold = threshold
        self.hold_s = hold_s
        self.require_wink = require_wink
        self.cooldown_s = cooldown_s
        self._closed_since: Optional[float] = None
        self._fired_this_closure = False
        self._cooldown_until = 0.0

    def update(self, right_ear: float, left_ear: float, now: float) -> BlinkState:
        state = BlinkState(right_closed=right_ear < self.threshold,
                           left_closed=left_ear < self.threshold)
        gesture = state.right_closed and (not self.require_wink or not state.left_closed)

        if not gesture:
            self._closed_since = None
            self._fired_this_closure = False
            return state

        if self._closed_since is None:
            self._closed_since = now
        held = now - self._closed_since
        state.hold_progress = min(1.0, held / self.hold_s)

        if (held >= self.hold_s and not self._fired_this_closure
                and now >= self._cooldown_until):
            self._fired_this_closure = True
            self._cooldown_until = now + self.cooldown_s
            state.fired = True
        return state


# --------------------------------------------------------------------------- #
class HeadScroller:
    """
    Converts the head-pitch offset from neutral into scroll events.

    offset < 0  -> head tilted UP   -> scroll up   (positive wheel)
    offset > 0  -> head tilted DOWN -> scroll down (negative wheel)

    Speed grows linearly with how far past the dead-zone the head is tilted,
    and events are rate-limited so scrolling is controllable.
    """

    MAX_GAIN = 4.0

    def __init__(self, deadzone: float, speed: float, interval_s: float,
                 invert: bool = False):
        self.deadzone = deadzone
        self.speed = speed
        self.interval_s = interval_s
        self.invert = invert
        self._last_emit = 0.0

    def update(self, offset: float, now: float) -> float:
        """Returns wheel notches to scroll now (0.0 if none)."""
        if abs(offset) <= self.deadzone:
            return 0.0
        if now - self._last_emit < self.interval_s:
            return 0.0
        self._last_emit = now
        gain = min(self.MAX_GAIN, 1.0 + (abs(offset) - self.deadzone) / self.deadzone)
        direction = 1.0 if offset < 0 else -1.0
        if self.invert:
            direction = -direction
        return direction * self.speed * gain

    def is_active(self, offset: float) -> bool:
        return abs(offset) > self.deadzone
