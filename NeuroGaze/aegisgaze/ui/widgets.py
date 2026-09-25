"""Small custom Tk widgets and the shared colour palette."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

# ---- Palette (dark, high-contrast for low-vision friendliness) ------------ #
BG = "#11151c"
CARD = "#1a202b"
CARD_BORDER = "#262e3d"
TEXT = "#e6ebf2"
MUTED = "#8b96a8"
ACCENT = "#3fb6ff"
OK = "#35d07f"
WARN = "#ffb347"
DANGER = "#ff5d5d"
FONT = "Segoe UI"


class ToggleSwitch(tk.Canvas):
    """iOS-style on/off switch. ``command(bool)`` fires on user clicks only."""

    W, H = 74, 36

    def __init__(self, master, command: Optional[Callable[[bool], None]] = None,
                 bg: str = CARD, **kwargs):
        super().__init__(master, width=self.W, height=self.H, bg=bg,
                         highlightthickness=0, cursor="hand2", **kwargs)
        self._value = False
        self._command = command
        self.bind("<Button-1>", self._on_click)
        self.bind("<space>", self._on_click)
        self._draw()

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        if bool(value) != self._value:
            self._value = bool(value)
            self._draw()

    def _on_click(self, _event=None) -> None:
        self.set(not self._value)
        if self._command:
            self._command(self._value)

    def _draw(self) -> None:
        self.delete("all")
        w, h, r = self.W, self.H, self.H // 2
        color = OK if self._value else "#3a4354"
        # Rounded track = two circles + a rectangle.
        self.create_oval(0, 0, h, h, fill=color, outline=color)
        self.create_oval(w - h, 0, w, h, fill=color, outline=color)
        self.create_rectangle(r, 0, w - r, h, fill=color, outline=color)
        knob_x = w - r if self._value else r
        self.create_oval(knob_x - r + 4, 4, knob_x + r - 4, h - 4,
                         fill="white", outline="")


class ProgressRing(tk.Canvas):
    """Circular progress indicator with a caption in the middle."""

    def __init__(self, master, size: int = 70, color: str = ACCENT,
                 label: str = "", bg: str = CARD):
        super().__init__(master, width=size, height=size, bg=bg, highlightthickness=0)
        self.size, self.color, self.label = size, color, label
        self.set(0.0)

    def set(self, progress: float) -> None:
        progress = max(0.0, min(1.0, progress))
        s, pad = self.size, 6
        self.delete("all")
        self.create_oval(pad, pad, s - pad, s - pad, outline="#2c3445", width=6)
        if progress > 0:
            self.create_arc(pad, pad, s - pad, s - pad, start=90,
                            extent=-359.9 * progress, style="arc",
                            outline=self.color, width=6)
        self.create_text(s / 2, s / 2, text=self.label, fill=TEXT,
                         font=(FONT, 8, "bold"))


class LevelBar(tk.Canvas):
    """
    Horizontal bar for a live value with an optional threshold marker
    (used for EAR and head-pitch readouts).
    """

    def __init__(self, master, width: int = 200, height: int = 12,
                 vmin: float = 0.0, vmax: float = 1.0, centered: bool = False,
                 bg: str = CARD):
        super().__init__(master, width=width, height=height, bg=bg, highlightthickness=0)
        self.w, self.h = width, height
        self.vmin, self.vmax, self.centered = vmin, vmax, centered

    def _x(self, v: float) -> float:
        v = max(self.vmin, min(self.vmax, v))
        return (v - self.vmin) / (self.vmax - self.vmin) * self.w

    def set(self, value: float, threshold: Optional[float] = None,
            color: str = ACCENT, band: Optional[float] = None) -> None:
        self.delete("all")
        self.create_rectangle(0, 0, self.w, self.h, fill="#2c3445", outline="")
        if band is not None:  # symmetric dead-zone band around zero
            self.create_rectangle(self._x(-band), 0, self._x(band), self.h,
                                  fill="#384257", outline="")
        x0 = self._x(0.0) if self.centered else 0
        self.create_rectangle(min(x0, self._x(value)), 2, max(x0, self._x(value)),
                              self.h - 2, fill=color, outline="")
        if threshold is not None:
            tx = self._x(threshold)
            self.create_line(tx, 0, tx, self.h, fill=WARN, width=2)
