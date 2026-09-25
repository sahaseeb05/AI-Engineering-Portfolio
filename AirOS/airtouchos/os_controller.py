"""
OSController: turns ``GestureOutput`` intents into real mouse/keyboard/volume
actions with PyAutoGUI (and pycaw for absolute volume on Windows).

Responsibilities:
    * Map the camera "active region" onto the full screen (edge padding).
    * Adaptive exponential-moving-average smoothing of the cursor.
    * Custom bounds checking (PyAutoGUI's corner fail-safe is disabled).
    * Mouse-button bookkeeping so a button is never left stuck down.
    * Guarded Alt+F4 that refuses to target the desktop, taskbar or this app.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pyautogui

from .config import AppConfig
from .gesture_engine import GestureOutput

log = logging.getLogger(__name__)

# PyAutoGUI's fail-safe aborts when the cursor hits a screen corner, which a
# gesture controller must be able to reach.  We replace it with our own bounds
# checking (see ``OSController._safe_point``) and the keyboard kill switch.
pyautogui.FAILSAFE = False
# PyAutoGUI sleeps 0.1 s after *every* call by default; that would cap the
# control loop at ~10 FPS.
pyautogui.PAUSE = 0.0
pyautogui.MINIMUM_DURATION = 0.0


class VolumeBackend:
    """System volume control.

    Prefers pycaw (Windows Core Audio) for true absolute levels.  Falls back to
    media volume keys, in which case the requested level is interpreted as a
    rate control: above ``volume_key_up_zone`` -> repeatedly press Volume Up,
    below ``volume_key_down_zone`` -> Volume Down, in between -> hold.
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._endpoint = None
        self._last_level: Optional[float] = None
        self._last_key_time = 0.0
        if sys.platform == "win32":
            self._endpoint = self._init_pycaw()
        self.mode = "absolute (pycaw)" if self._endpoint is not None else "relative (media keys)"
        log.info("Volume backend: %s", self.mode)

    @staticmethod
    def _init_pycaw():
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        except ImportError:
            log.warning("pycaw not installed; falling back to volume keys.")
            return None
        try:
            speakers = AudioUtilities.GetSpeakers()
            # pycaw >= 20240316 wraps the device and exposes EndpointVolume.
            endpoint = getattr(speakers, "EndpointVolume", None)
            if endpoint is None:
                from ctypes import POINTER, cast
                from comtypes import CLSCTX_ALL

                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            endpoint.GetMasterVolumeLevelScalar()  # probe once
            return endpoint
        except Exception as exc:  # COM errors are not a single exception type
            log.warning("pycaw could not open the default speaker (%s); using volume keys.", exc)
            return None

    def current_level(self) -> Optional[float]:
        """Current master volume in [0, 1], or None if unknown."""
        if self._endpoint is None:
            return None
        try:
            return float(self._endpoint.GetMasterVolumeLevelScalar())
        except Exception:
            return None

    def apply(self, level: float) -> None:
        """Drive the system volume toward *level* in [0, 1]."""
        level = float(np.clip(level, 0.0, 1.0))
        if self._endpoint is not None:
            if self._last_level is None or abs(level - self._last_level) >= 1e-3:
                try:
                    self._endpoint.SetMasterVolumeLevelScalar(level, None)
                    self._last_level = level
                except Exception as exc:
                    log.error("Setting volume failed: %s", exc)
            return

        now = time.perf_counter()
        if now - self._last_key_time < self._cfg.volume_key_interval_s:
            return
        if level >= self._cfg.volume_key_up_zone:
            pyautogui.press("volumeup")
            self._last_key_time = now
        elif level <= self._cfg.volume_key_down_zone:
            pyautogui.press("volumedown")
            self._last_key_time = now

    def reset(self) -> None:
        self._last_level = None


class OSController:
    """Executes gesture intents against the operating system."""

    # Window classes that must never receive Alt+F4 (it would open the Windows
    # shutdown dialog or kill Explorer's taskbar).
    _PROTECTED_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self.screen_w, self.screen_h = pyautogui.size()
        self._smoothed: Optional[np.ndarray] = None
        self._last_raw: Optional[np.ndarray] = None
        self._mouse_down = False
        self.volume = VolumeBackend(config)
        self.last_blocked_reason = ""
        log.info("Screen %dx%d", self.screen_w, self.screen_h)

    # ---------------------------------------------------------------- mapping
    def map_to_screen(self, nx: float, ny: float) -> np.ndarray:
        """Map a normalised camera point to screen pixels.

        The inner rectangle [margin, 1 - margin] of the camera frame is
        stretched to the whole screen; points in the margin clamp to the edge.
        This lets the fingertip reach all four corners comfortably while the
        hand is still fully visible to the tracker.
        """
        mx, my = self._cfg.margin_x, self._cfg.margin_y
        x = np.interp(nx, [mx, 1.0 - mx], [0.0, self.screen_w - 1.0])
        y = np.interp(ny, [my, 1.0 - my], [0.0, self.screen_h - 1.0])
        return np.array([x, y], dtype=np.float64)

    def _safe_point(self, point: np.ndarray) -> Optional[Tuple[int, int]]:
        """Custom bounds check replacing PyAutoGUI's fail-safe.

        Rejects NaN/inf and clamps to the visible primary screen so the cursor
        can sit exactly on a corner pixel but never be sent off-screen.
        """
        if not np.all(np.isfinite(point)):
            return None
        x = int(round(min(max(point[0], 0.0), self.screen_w - 1)))
        y = int(round(min(max(point[1], 0.0), self.screen_h - 1)))
        return x, y

    def _smooth(self, raw: np.ndarray, precision: bool) -> np.ndarray:
        """Adaptive EMA: s_t = a * x_t + (1 - a) * s_{t-1}.

        ``a`` rises with fingertip speed (responsive sweeps) and falls for slow
        motion (steady aiming).  During a forming pinch it is scaled down
        further so the act of pinching does not drag the cursor off target.
        """
        if self._smoothed is None or self._last_raw is None:
            self._smoothed = raw.copy()
            self._last_raw = raw.copy()
            return self._smoothed
        cfg = self._cfg
        speed = float(np.linalg.norm(raw - self._last_raw))
        self._last_raw = raw.copy()
        alpha = float(np.interp(speed, [cfg.ema_speed_low_px, cfg.ema_speed_high_px],
                                [cfg.ema_alpha_min, cfg.ema_alpha_max]))
        if precision:
            alpha *= cfg.precision_alpha_factor
        self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed
        return self._smoothed

    def reset_smoothing(self) -> None:
        """Forget filter history (call when the hand re-appears) so the cursor
        does not glide across the screen from a stale position."""
        self._smoothed = None
        self._last_raw = None

    # ---------------------------------------------------------------- actions
    def move_cursor(self, nx: float, ny: float, precision: bool = False) -> None:
        target = self._smooth(self.map_to_screen(nx, ny), precision)
        point = self._safe_point(target)
        if point is None:
            return
        cur_x, cur_y = pyautogui.position()
        if math.hypot(point[0] - cur_x, point[1] - cur_y) < self._cfg.cursor_deadzone_px:
            return
        pyautogui.moveTo(point[0], point[1], _pause=False)

    def left_click(self) -> None:
        pyautogui.click(button="left", _pause=False)

    def right_click(self) -> None:
        pyautogui.click(button="right", _pause=False)

    def drag_start(self) -> None:
        if not self._mouse_down:
            pyautogui.mouseDown(button="left", _pause=False)
            self._mouse_down = True

    def drag_end(self) -> None:
        if self._mouse_down:
            pyautogui.mouseUp(button="left", _pause=False)
            self._mouse_down = False

    def close_window(self) -> bool:
        """Send Alt+F4 to the foreground window unless it is protected.

        Returns True if the shortcut was sent.
        """
        allowed, reason = self._close_target_allowed()
        if not allowed:
            self.last_blocked_reason = reason
            log.info("Alt+F4 blocked: %s", reason)
            return False
        self.last_blocked_reason = ""
        self.drag_end()                     # never close with a button held
        pyautogui.hotkey("alt", "f4")
        log.info("Alt+F4 sent.")
        return True

    def _close_target_allowed(self) -> Tuple[bool, str]:
        if not self._cfg.protect_desktop_from_close or sys.platform != "win32":
            return True, ""
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False, "no foreground window"
        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        length = user32.GetWindowTextLengthW(hwnd)
        title_buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buf, length + 1)
        cls, title = class_buf.value, title_buf.value

        if cls in self._PROTECTED_CLASSES:
            return False, "desktop/taskbar is focused (would open shutdown dialog)"
        if title == self._cfg.window_name:
            return False, "the AirTouchOS preview is focused"
        console = ctypes.windll.kernel32.GetConsoleWindow()
        if console and hwnd == console:
            return False, "the AirTouchOS console is focused"
        return True, ""

    def release_all(self) -> None:
        """Release any held mouse button (pause, shutdown, error paths)."""
        try:
            self.drag_end()
        except Exception as exc:
            log.error("Failed to release mouse button: %s", exc)
            self._mouse_down = False

    # --------------------------------------------------------------- dispatch
    def apply(self, out: GestureOutput) -> None:
        """Execute every intent in *out*.

        Order matters: button events are issued *before* the cursor moves so
        that a click lands where the pinch was made, and drag_end is issued
        before anything that could change focus.
        """
        if out.drag_end:
            self.drag_end()
        if out.left_click:
            self.left_click()
        if out.right_click:
            self.right_click()
        if out.close_window:
            self.close_window()
        if out.drag_start:
            self.drag_start()
        if out.volume_level is not None:
            self.volume.apply(out.volume_level)
        else:
            self.volume.reset()
        if out.cursor_target is not None and not out.paused:
            self.move_cursor(out.cursor_target[0], out.cursor_target[1], out.precision)
