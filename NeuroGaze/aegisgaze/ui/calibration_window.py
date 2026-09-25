"""
Full-screen calibration UI.

Displays one target at a time (centre, then the four corners). Each target
first *settles* (ring shrinking onto the dot: "move your eyes here") and then
*samples* (dot fills: "hold still"). Sampling itself happens in the engine
thread; this window only visualises the session's state.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from ..calibration import CalibrationSession
from .widgets import ACCENT, FONT, MUTED, OK, TEXT


class CalibrationWindow:
    POLL_MS = 30

    def __init__(self, root: tk.Tk, session: CalibrationSession,
                 on_cancel: Callable[[], None]):
        self.session = session
        self.on_cancel = on_cancel
        self.win = tk.Toplevel(root)
        self.win.configure(bg="black")
        self.win.attributes("-fullscreen", True)
        self.win.attributes("-topmost", True)
        self.win.config(cursor="none")
        self.canvas = tk.Canvas(self.win, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.win.bind("<Escape>", lambda _e: self.cancel())
        self.win.focus_force()
        self._closed = False
        self._tick()

    @property
    def closed(self) -> bool:
        return self._closed

    def cancel(self) -> None:
        if not self._closed:
            self.on_cancel()
            self.close()

    def close(self) -> None:
        self._closed = True
        try:
            self.win.destroy()
        except tk.TclError:
            pass

    def _tick(self) -> None:
        if self._closed:
            return
        s = self.session
        if s.finished or s.cancelled:
            self.close()
            return

        target = s.current
        phase, progress = s.phase()
        w, h = self.win.winfo_width(), self.win.winfo_height()
        c = self.canvas
        c.delete("all")

        c.create_text(w / 2, h / 2 - 90 if target.name != "centre" else h / 2 - 140,
                      text="Keep your head still — follow the dot with your EYES only",
                      fill=TEXT, font=(FONT, 20, "bold"))
        c.create_text(w / 2, (h / 2 - 50) if target.name != "centre" else h / 2 - 100,
                      text=f"Target {s.index + 1} / {len(s.targets)}  ({target.name})"
                           "      ·      Esc to cancel",
                      fill=MUTED, font=(FONT, 13))

        x, y = target.screen_xy
        if phase == "settle":
            # Shrinking ring draws the eye onto the target.
            r = 60 - 45 * progress
            c.create_oval(x - r, y - r, x + r, y + r, outline=ACCENT, width=3)
            c.create_oval(x - 8, y - 8, x + 8, y + 8, fill=ACCENT, outline="")
        else:
            # Filling ring: samples being collected.
            c.create_oval(x - 18, y - 18, x + 18, y + 18, outline="#444", width=4)
            c.create_arc(x - 18, y - 18, x + 18, y + 18, start=90,
                         extent=-359.9 * progress, style="arc", outline=OK, width=4)
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white", outline="")

        self.win.after(self.POLL_MS, self._tick)
