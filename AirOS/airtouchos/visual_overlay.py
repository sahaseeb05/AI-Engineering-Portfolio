"""
VisualOverlay: all OpenCV drawing for the preview window.

Draws the hand skeleton, the cursor active region, pinch connector lines
(colour-coded by state), hold-progress rings for the fist and pause gestures,
a volume bar, the status/FPS panel, transient event banners and an optional
live calibration read-out of every normalised metric.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import AppConfig
from .gesture_engine import FINGER_NAMES, Gesture, GestureOutput
from .hand_tracker import (
    HAND_CONNECTIONS,
    HandObservation,
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
)

# BGR colours.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
GREEN = (80, 220, 80)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)
RED = (60, 60, 235)
CYAN = (230, 200, 40)
MAGENTA = (200, 80, 220)

FONT = cv2.FONT_HERSHEY_SIMPLEX

_GESTURE_COLOURS = {
    Gesture.NO_HAND: GREY,
    Gesture.POINTER: GREEN,
    Gesture.PINCH: YELLOW,
    Gesture.DRAG: ORANGE,
    Gesture.RIGHT_PINCH: MAGENTA,
    Gesture.FIST: RED,
    Gesture.VOLUME: CYAN,
    Gesture.OPEN_PALM: WHITE,
    Gesture.PAUSED: RED,
}


class FpsCounter:
    """Exponentially smoothed frames-per-second estimate."""

    def __init__(self, smoothing: float = 0.1) -> None:
        self._smoothing = smoothing
        self._last: Optional[float] = None
        self.fps = 0.0

    def tick(self, now: Optional[float] = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else (
                    self._smoothing * inst + (1 - self._smoothing) * self.fps)
        self._last = now
        return self.fps


class VisualOverlay:
    """Renders annotations onto the (mirrored) camera frame in place."""

    EVENT_DISPLAY_S = 1.2

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self.fps = FpsCounter()
        self.show_calibration = config.show_calibration
        self._event_text = ""
        self._event_time = -1e9

    # ----------------------------------------------------------------- public
    def flash(self, text: str) -> None:
        """Show *text* as a banner for a short time."""
        self._event_text = text
        self._event_time = time.perf_counter()

    def draw(
        self,
        frame: np.ndarray,
        obs: Optional[HandObservation],
        out: GestureOutput,
        volume_mode: str,
        current_volume: Optional[float],
        blocked_reason: str = "",
    ) -> np.ndarray:
        """Draw every annotation for this frame and return *frame*."""
        fps = self.fps.tick()
        for event in out.events:
            self.flash(event)
        if blocked_reason and out.close_window:
            self.flash(f"CLOSE BLOCKED: {blocked_reason}")

        self._draw_active_region(frame)
        if obs is not None:
            self._draw_skeleton(frame, obs, out)
            self._draw_pinch_lines(frame, obs, out)
            self._draw_progress_rings(frame, obs, out)
        if out.gesture == Gesture.VOLUME or out.volume_level is not None:
            self._draw_volume_bar(frame, out.volume_level, current_volume, volume_mode)
        if out.paused:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 6)
        self._draw_status_panel(frame, out, fps)
        if self.show_calibration and out.metrics is not None:
            self._draw_calibration(frame, out)
        self._draw_event_banner(frame)
        self._draw_help(frame)
        return frame

    # ------------------------------------------------------------ primitives
    @staticmethod
    def _text(frame, text, org, scale=0.55, colour=WHITE, thickness=1) -> None:
        """Text with a dark outline so it stays readable on any background."""
        cv2.putText(frame, text, org, FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)

    @staticmethod
    def _panel(frame, top_left, bottom_right, alpha=0.55) -> None:
        """Translucent dark rectangle behind text."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)

    @staticmethod
    def _pt(obs: HandObservation, idx: int) -> Tuple[int, int]:
        x, y = obs.landmarks_px[idx]
        return int(round(x)), int(round(y))

    # --------------------------------------------------------------- layers
    def _draw_active_region(self, frame: np.ndarray) -> None:
        """Rectangle that maps to the full screen (see margin_x / margin_y)."""
        h, w = frame.shape[:2]
        x1, y1 = int(w * self._cfg.margin_x), int(h * self._cfg.margin_y)
        x2, y2 = int(w * (1 - self._cfg.margin_x)), int(h * (1 - self._cfg.margin_y))
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1, cv2.LINE_AA)
        self._text(frame, "screen area", (x1 + 4, y2 - 6), 0.4, GREY)

    def _draw_skeleton(self, frame: np.ndarray, obs: HandObservation, out: GestureOutput) -> None:
        colour = _GESTURE_COLOURS.get(out.gesture, WHITE)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, self._pt(obs, a), self._pt(obs, b), colour, 2, cv2.LINE_AA)
        for i in range(21):
            radius = 6 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, self._pt(obs, i), radius, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, self._pt(obs, i), radius, colour, 1, cv2.LINE_AA)
        # Highlight the pointer fingertip.
        cv2.circle(frame, self._pt(obs, INDEX_TIP), 11, colour, 2, cv2.LINE_AA)

    def _pinch_colour(self, distance: float, on: float, active: bool) -> Tuple[int, int, int]:
        if active:
            return RED
        if distance < on * self._cfg.precision_zone_factor:
            return YELLOW
        return GREEN

    def _draw_pinch_lines(self, frame: np.ndarray, obs: HandObservation, out: GestureOutput) -> None:
        m = out.metrics
        if m is None:
            return
        thumb = self._pt(obs, THUMB_TIP)
        index = self._pt(obs, INDEX_TIP)
        middle = self._pt(obs, MIDDLE_TIP)

        if out.gesture == Gesture.VOLUME:
            # Show the vertical component that is being measured.
            cv2.line(frame, thumb, index, CYAN, 2, cv2.LINE_AA)
            cv2.line(frame, (index[0], thumb[1]), index, CYAN, 1, cv2.LINE_AA)
            cv2.line(frame, thumb, (index[0], thumb[1]), CYAN, 1, cv2.LINE_AA)
            return

        left_colour = self._pinch_colour(m.left_pinch, self._cfg.left_pinch_on, out.left_pinch_active)
        cv2.line(frame, thumb, index, left_colour, 2, cv2.LINE_AA)
        mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
        cv2.circle(frame, mid, 7 if out.left_pinch_active else 4, left_colour, -1, cv2.LINE_AA)
        self._text(frame, f"L {m.left_pinch:.2f}", (mid[0] + 8, mid[1] - 4), 0.4, left_colour)

        right_colour = self._pinch_colour(m.right_pinch, self._cfg.right_pinch_on, out.right_pinch_active)
        cv2.line(frame, thumb, middle, right_colour, 1, cv2.LINE_AA)
        mid_r = ((thumb[0] + middle[0]) // 2, (thumb[1] + middle[1]) // 2)
        cv2.circle(frame, mid_r, 6 if out.right_pinch_active else 3, right_colour, -1, cv2.LINE_AA)
        self._text(frame, f"R {m.right_pinch:.2f}", (mid_r[0] + 8, mid_r[1] - 4), 0.4, right_colour)

    def _draw_progress_rings(self, frame: np.ndarray, obs: HandObservation, out: GestureOutput) -> None:
        if out.palm_progress > 0 and out.metrics is not None:
            h, w = frame.shape[:2]
            cx, cy = out.metrics.palm_center
            centre = (int(cx * max(w, h)), int(cy * max(w, h)))
            radius = int(out.metrics.scale_px * 0.9)
            self._ring(frame, centre, radius, out.palm_progress, WHITE)
            label = "RESUME" if out.paused else "PAUSE"
            self._text(frame, f"{label} {out.palm_progress * 100:.0f}%",
                       (centre[0] - 45, centre[1] - radius - 10), 0.5, WHITE)
        if out.gesture == Gesture.FIST:
            centre = self._pt(obs, WRIST)
            radius = int(out.metrics.scale_px * 0.6) if out.metrics else 30
            self._ring(frame, centre, radius, out.fist_progress, RED)

    @staticmethod
    def _ring(frame, centre, radius, progress, colour) -> None:
        radius = max(radius, 12)
        cv2.circle(frame, centre, radius, GREY, 2, cv2.LINE_AA)
        cv2.ellipse(frame, centre, (radius, radius), -90, 0, 360 * min(progress, 1.0),
                    colour, 4, cv2.LINE_AA)

    def _draw_volume_bar(self, frame, level, current, mode) -> None:
        h, w = frame.shape[:2]
        x1, x2 = w - 50, w - 25
        y1, y2 = 90, h - 90
        self._panel(frame, (x1 - 10, y1 - 30), (x2 + 10, y2 + 40))
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1)
        shown = current if current is not None else level
        if shown is not None:
            fill_top = int(y2 - (y2 - y1) * shown)
            cv2.rectangle(frame, (x1 + 2, fill_top), (x2 - 2, y2 - 2), CYAN, -1)
            self._text(frame, f"{shown * 100:.0f}%", (x1 - 8, y2 + 22), 0.5, CYAN)
        if level is not None and current is None:
            # Relative key mode: draw the up/down zones instead of an absolute level.
            for zone in (self._cfg.volume_key_up_zone, self._cfg.volume_key_down_zone):
                zy = int(y2 - (y2 - y1) * zone)
                cv2.line(frame, (x1 - 5, zy), (x2 + 5, zy), YELLOW, 1)
        self._text(frame, "VOL", (x1 - 2, y1 - 10), 0.5, CYAN)
        self._text(frame, mode, (10, h - 38), 0.4, CYAN)

    def _draw_status_panel(self, frame: np.ndarray, out: GestureOutput, fps: float) -> None:
        self._panel(frame, (0, 0), (250, 78))
        colour = _GESTURE_COLOURS.get(out.gesture, WHITE)
        self._text(frame, f"Gesture: {out.gesture.value}", (10, 24), 0.6, colour, 2)
        fps_colour = GREEN if fps >= 20 else (YELLOW if fps >= 12 else RED)
        self._text(frame, f"FPS: {fps:5.1f}", (10, 48), 0.55, fps_colour)
        state, state_colour = ("PAUSED", RED) if out.paused else ("ACTIVE", GREEN)
        self._text(frame, f"Mouse: {state}", (10, 70), 0.55, state_colour)

    def _draw_calibration(self, frame: np.ndarray, out: GestureOutput) -> None:
        m = out.metrics
        cfg = self._cfg
        x, y0 = 10, 100
        lines = [
            ("CALIBRATION (units = palm length)", WHITE),
            (f"L pinch {m.left_pinch:4.2f} on<{cfg.left_pinch_on:.2f} off>{cfg.left_pinch_off:.2f}",
             RED if m.left_pinch < cfg.left_pinch_on else WHITE),
            (f"R pinch {m.right_pinch:4.2f} on<{cfg.right_pinch_on:.2f} off>{cfg.right_pinch_off:.2f}",
             RED if m.right_pinch < cfg.right_pinch_on else WHITE),
            (f"vol span {m.volume_span:4.2f} [{cfg.volume_span_min:.2f}..{cfg.volume_span_max:.2f}]", WHITE),
            (f"fist if tips<{cfg.fist_tip_ratio:.2f} & thumb<{cfg.fist_thumb_ratio:.2f}", WHITE),
        ]
        for name, ratio, ext in zip(FINGER_NAMES, m.tip_to_wrist, m.extended):
            limit = cfg.fist_thumb_ratio if name == "thumb" else cfg.fist_tip_ratio
            colour = RED if ratio < limit else WHITE    # red = satisfies fist test
            lines.append((f" {name:<6} tip-wrist {ratio:4.2f} {'EXT' if ext else 'fold'}", colour))
        self._panel(frame, (0, y0 - 16), (300, y0 + len(lines) * 18))
        for i, (text, colour) in enumerate(lines):
            self._text(frame, text, (x, y0 + i * 18), 0.42, colour)

    def _draw_event_banner(self, frame: np.ndarray) -> None:
        if not self._event_text:
            return
        age = time.perf_counter() - self._event_time
        if age > self.EVENT_DISPLAY_S:
            return
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(self._event_text, FONT, 0.8, 2)
        x, y = (w - tw) // 2, 115
        self._panel(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), 0.65)
        self._text(frame, self._event_text, (x, y), 0.8, YELLOW, 2)

    def _draw_help(self, frame: np.ndarray) -> None:
        h = frame.shape[0]
        self._text(frame, "[q/Esc] quit  [p] pause  [c] calibration", (10, h - 14), 0.45, GREY)
