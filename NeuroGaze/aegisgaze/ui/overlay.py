"""
On-screen dwell indicator that follows the cursor.

A small, borderless, always-on-top window draws a circular "loading" ring
around the cursor while a dwell click (blue) or a blink right-click (orange)
is charging. The ring's radius equals the dwell tolerance radius, so the user
sees exactly how still they need to hold their gaze.

IMPORTANT: the synthetic click lands exactly under this window, so it MUST be
click-through. On Windows we combine a colour-keyed transparent background
with the WS_EX_TRANSPARENT extended style. Other platforms cannot guarantee
click-through from Tk, so the overlay is disabled there and progress is shown
in the dashboard only.
"""

from __future__ import annotations

import logging
import platform
import tkinter as tk

from .widgets import ACCENT, WARN

log = logging.getLogger(__name__)

_KEY = "#ff00fe"  # colour rendered fully transparent


class DwellOverlay:
    def __init__(self, root: tk.Tk, radius: float = 30.0):
        self.available = platform.system() == "Windows"
        self._visible = False
        self._radius = radius
        if not self.available:
            return

        self._size = int(radius * 2 + 16)
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-transparentcolor", _KEY)
        self.win.configure(bg=_KEY)
        self.canvas = tk.Canvas(self.win, bg=_KEY, highlightthickness=0,
                                width=self._size, height=self._size)
        self.canvas.pack()
        self.win.withdraw()
        self.win.update_idletasks()
        self._make_click_through()

    def _make_click_through(self) -> None:
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x00080000, 0x00000020
            WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x00000080, 0x08000000
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED |
                                  WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        except Exception as exc:
            log.warning("Overlay click-through unavailable (%s); disabling overlay.", exc)
            self.available = False
            self.win.destroy()

    # ------------------------------------------------------------------ #
    def update(self, cursor, dwell: float, blink: float, radius: float) -> None:
        """Redraw at ``cursor``; hides itself when nothing is charging."""
        if not self.available:
            return
        if cursor is None or (dwell <= 0.02 and blink <= 0.02):
            self.hide()
            return

        if abs(radius - self._radius) > 0.5:          # radius changed in settings
            self._radius = radius
            self._size = int(radius * 2 + 16)
            self.canvas.config(width=self._size, height=self._size)

        s, r = self._size, self._radius
        c = s / 2
        self.canvas.delete("all")
        # Faint tolerance circle + progress arcs (clockwise from 12 o'clock).
        self.canvas.create_oval(c - r, c - r, c + r, c + r, outline="#5a6478", width=2)
        if dwell > 0.02:
            self.canvas.create_arc(c - r, c - r, c + r, c + r, start=90,
                                   extent=-359.9 * min(dwell, 1.0), style="arc",
                                   outline=ACCENT, width=5)
        if blink > 0.02:
            ri = r - 8
            self.canvas.create_arc(c - ri, c - ri, c + ri, c + ri, start=90,
                                   extent=-359.9 * min(blink, 1.0), style="arc",
                                   outline=WARN, width=4)

        x, y = int(cursor[0] - s / 2), int(cursor[1] - s / 2)
        self.win.geometry(f"{s}x{s}+{x}+{y}")
        if not self._visible:
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self._visible = True

    def hide(self) -> None:
        if self.available and self._visible:
            self.win.withdraw()
            self._visible = False
