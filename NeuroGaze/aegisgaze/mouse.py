"""
Thin, thread-safe wrapper around PyAutoGUI.

All OS-level input goes through here so there is exactly one place that
checks the global enable flag – the panic switch is therefore guaranteed to
stop every action (move, click, scroll) instantly.
"""

from __future__ import annotations

import logging
import platform
import threading

import pyautogui

log = logging.getLogger(__name__)

# PyAutoGUI sleeps 0.1 s after EVERY call by default, which would cap the
# cursor at ~10 updates/s and add visible lag. We pace ourselves instead.
pyautogui.PAUSE = 0
# Keep the corner fail-safe as an extra emergency stop; we clamp the cursor
# 1 px inside the screen so gaze control itself never triggers it.
pyautogui.FAILSAFE = True

_IS_WINDOWS = platform.system() == "Windows"
_WHEEL_DELTA = 120  # one physical wheel notch on Windows


class MouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self._enabled = False
        self._lock = threading.Lock()
        self._scroll_remainder = 0.0

    # ---- Global enable flag (panic switch) ----------------------------- #
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = bool(value)
            self._scroll_remainder = 0.0
        log.info("Mouse control %s", "ENABLED" if value else "DISABLED")

    # ---- Actions ------------------------------------------------------- #
    def clamp(self, x: float, y: float) -> tuple[int, int]:
        """Clamp to the visible screen, 1 px away from fail-safe corners."""
        return (int(min(max(x, 1), self.screen_w - 2)),
                int(min(max(y, 1), self.screen_h - 2)))

    def move(self, x: float, y: float) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._safe(pyautogui.moveTo, *self.clamp(x, y), _pause=False)

    def left_click(self) -> None:
        with self._lock:
            if self._enabled:
                self._safe(pyautogui.click)

    def right_click(self) -> None:
        with self._lock:
            if self._enabled:
                self._safe(pyautogui.rightClick)

    def scroll(self, notches: float) -> None:
        """Scroll by (possibly fractional) wheel notches; + = up."""
        with self._lock:
            if not self._enabled or notches == 0:
                return
            if _IS_WINDOWS:
                # Windows accepts raw wheel deltas -> smooth fractional scroll.
                self._safe(pyautogui.scroll, int(notches * _WHEEL_DELTA))
            else:
                # Other platforms only take whole clicks; accumulate the rest.
                self._scroll_remainder += notches
                whole = int(self._scroll_remainder)
                if whole:
                    self._scroll_remainder -= whole
                    self._safe(pyautogui.scroll, whole)

    @staticmethod
    def _safe(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except pyautogui.FailSafeException:
            # User slammed the physical mouse into a corner: respect it.
            log.warning("PyAutoGUI fail-safe triggered.")
            raise
        except Exception as exc:  # e.g. secure desktop / UAC prompt on Windows
            log.debug("Input injection failed: %s", exc)
