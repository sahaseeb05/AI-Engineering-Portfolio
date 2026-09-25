"""
Main desktop dashboard (Tkinter).

Layout
------
+-------------------------------+----------------------------+
|  Live camera preview          |  Mouse control  [ switch ] |
|  (mesh, eye contours, irises) |  [Calibrate] [Re-centre]   |
|                               |  ┌ Status ┬ Settings ┐     |
|                               |  │ ...    │           │     |
+-------------------------------+----------------------------+
|  message banner                                            |
+------------------------------------------------------------+

The dashboard polls ``Engine.snapshot()`` every ~30 ms; it never blocks on
the vision pipeline.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .. import config as C
from ..engine import Engine
from ..hotkeys import HotkeyListener
from .calibration_window import CalibrationWindow
from .overlay import DwellOverlay
from .widgets import (ACCENT, BG, CARD, CARD_BORDER, DANGER, FONT, MUTED, OK,
                      TEXT, WARN, LevelBar, ProgressRing, ToggleSwitch)

log = logging.getLogger(__name__)

PREVIEW_W, PREVIEW_H = 640, 480


class Dashboard:
    POLL_MS = 30

    def __init__(self, root: tk.Tk, engine: Engine, settings: C.Settings):
        self.root = root
        self.engine = engine
        self.settings = settings
        self._photo = None
        self._last_preview_id = None
        self._calib_window: CalibrationWindow | None = None
        self._last_message = ""

        root.title("AegisGaze — Eye-Gaze & Head-Pose Laptop Controller")
        root.configure(bg=BG)
        root.minsize(1100, 640)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._init_styles()
        self._build()

        self.overlay = DwellOverlay(root, settings.dwell_radius_px)

        # Global hotkeys (Esc panic / F8 toggle). If the global hook is not
        # available, fall back to window-local bindings.
        self.hotkeys = HotkeyListener(settings.panic_key, settings.toggle_key,
                                      on_panic=engine.panic,
                                      on_toggle=engine.toggle_control)
        if not self.hotkeys.start():
            root.bind(f"<{settings.toggle_key.upper()}>", lambda _e: engine.toggle_control())
        root.bind("<Escape>", lambda _e: engine.panic())   # idempotent; harmless twice

        self.root.after(self.POLL_MS, self._poll)

    # ================================================================== #
    # Construction
    # ================================================================== #
    def _init_styles(self) -> None:
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure(".", background=CARD, foreground=TEXT, font=(FONT, 10))
        st.configure("TFrame", background=CARD)
        st.configure("Bg.TFrame", background=BG)
        st.configure("TLabel", background=CARD, foreground=TEXT)
        st.configure("Muted.TLabel", foreground=MUTED)
        st.configure("Head.TLabel", background=BG, foreground=TEXT, font=(FONT, 18, "bold"))
        st.configure("Sub.TLabel", background=BG, foreground=MUTED, font=(FONT, 10))
        st.configure("Big.TLabel", font=(FONT, 13, "bold"))
        st.configure("TCheckbutton", background=CARD, foreground=TEXT)
        st.map("TCheckbutton", background=[("active", CARD)])
        st.configure("Accent.TButton", background=ACCENT, foreground="#0b1420",
                     font=(FONT, 10, "bold"), padding=(12, 8), borderwidth=0)
        st.map("Accent.TButton", background=[("active", "#6cc8ff")])
        st.configure("TButton", background="#2c3445", foreground=TEXT,
                     padding=(12, 8), borderwidth=0)
        st.map("TButton", background=[("active", "#3a4459")])
        st.configure("TNotebook", background=CARD, borderwidth=0)
        st.configure("TNotebook.Tab", background="#232a37", foreground=MUTED,
                     padding=(14, 6))
        st.map("TNotebook.Tab", background=[("selected", CARD)],
               foreground=[("selected", TEXT)])
        st.configure("Horizontal.TScale", background=CARD, troughcolor="#2c3445")
        st.configure("TCombobox", fieldbackground="#2c3445", foreground=TEXT)

    def _card(self, parent, **pack) -> tk.Frame:
        f = tk.Frame(parent, bg=CARD, highlightbackground=CARD_BORDER,
                     highlightthickness=1, padx=14, pady=12)
        f.pack(**pack)
        return f

    def _build(self) -> None:
        # ---- Header ------------------------------------------------------ #
        header = ttk.Frame(self.root, style="Bg.TFrame", padding=(16, 12, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text="AegisGaze", style="Head.TLabel").pack(side="left")
        ttk.Label(header, text="   AI eye-gaze & head-pose laptop controller",
                  style="Sub.TLabel").pack(side="left", pady=(6, 0))
        self.status_pill = tk.Label(header, text="● STARTING", bg=BG, fg=MUTED,
                                    font=(FONT, 11, "bold"))
        self.status_pill.pack(side="right")

        body = ttk.Frame(self.root, style="Bg.TFrame", padding=(16, 8))
        body.pack(fill="both", expand=True)

        # ---- Camera preview --------------------------------------------- #
        left = self._card(body, side="left", anchor="n")
        # Fixed pixel-size frame so the layout doesn't jump between text/image.
        holder = tk.Frame(left, bg="black", width=PREVIEW_W, height=PREVIEW_H)
        holder.pack_propagate(False)
        holder.pack()
        self.preview = tk.Label(holder, bg="black", fg=MUTED, font=(FONT, 12),
                                text="Starting camera…", wraplength=520, justify="center")
        self.preview.pack(fill="both", expand=True)
        self.retry_btn = ttk.Button(left, text="Retry camera",
                                    command=lambda: self.engine.reconnect_camera(
                                        self.settings.camera_index))
        # (packed only when a camera error is shown)

        # ---- Right column ------------------------------------------------ #
        right = ttk.Frame(body, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        ctrl = self._card(right, fill="x")
        row = tk.Frame(ctrl, bg=CARD)
        row.pack(fill="x")
        tk.Label(row, text="Mouse control", bg=CARD, fg=TEXT,
                 font=(FONT, 14, "bold")).pack(side="left")
        self.switch = ToggleSwitch(row, command=self._on_switch)
        self.switch.pack(side="right")
        self.switch_label = tk.Label(row, text="OFF", bg=CARD, fg=MUTED,
                                     font=(FONT, 11, "bold"))
        self.switch_label.pack(side="right", padx=10)
        tk.Label(ctrl, text=f"{self.settings.panic_key.upper()} = instant panic stop    ·    "
                            f"{self.settings.toggle_key.upper()} = toggle on/off",
                 bg=CARD, fg=MUTED, font=(FONT, 9)).pack(anchor="w", pady=(6, 8))
        btns = tk.Frame(ctrl, bg=CARD)
        btns.pack(fill="x")
        ttk.Button(btns, text="🎯  Calibration Mode", style="Accent.TButton",
                   command=self.start_calibration).pack(side="left")
        ttk.Button(btns, text="Re-centre head", command=self.engine.recenter_head
                   ).pack(side="left", padx=8)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, pady=(12, 0))
        status_tab = ttk.Frame(nb, padding=14)
        settings_tab = ttk.Frame(nb, padding=14)
        nb.add(status_tab, text="Live status")
        nb.add(settings_tab, text="Settings")
        self._build_status(status_tab)
        self._build_settings(settings_tab)

        # ---- Banner ------------------------------------------------------ #
        self.banner = tk.Label(self.root, text="", bg=BG, fg=MUTED, anchor="w",
                               font=(FONT, 10), padx=18, pady=6)
        self.banner.pack(fill="x")

    def _build_status(self, tab) -> None:
        rings = ttk.Frame(tab)
        rings.pack(fill="x")
        self.dwell_ring = ProgressRing(rings, 76, ACCENT, "DWELL")
        self.dwell_ring.pack(side="left")
        self.blink_ring = ProgressRing(rings, 76, WARN, "R-BLINK")
        self.blink_ring.pack(side="left", padx=10)
        info = ttk.Frame(rings)
        info.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.lbl_face = ttk.Label(info, text="Face: —", style="Big.TLabel")
        self.lbl_face.pack(anchor="w")
        self.lbl_calib = ttk.Label(info, text="Calibration: —")
        self.lbl_calib.pack(anchor="w")
        self.lbl_action = ttk.Label(info, text="Last action: —", style="Muted.TLabel")
        self.lbl_action.pack(anchor="w")

        grid = ttk.Frame(tab)
        grid.pack(fill="x", pady=(16, 0))
        self.bars = {}
        for r, (key, label, kw) in enumerate([
            ("rear", "Right eye EAR", dict(vmin=0.0, vmax=0.45)),
            ("lear", "Left eye EAR", dict(vmin=0.0, vmax=0.45)),
            ("pitch", "Head pitch", dict(vmin=-0.15, vmax=0.15, centered=True)),
        ]):
            ttk.Label(grid, text=label, width=14).grid(row=r, column=0, sticky="w", pady=4)
            bar = LevelBar(grid, width=230, **kw)
            bar.grid(row=r, column=1, sticky="w")
            val = ttk.Label(grid, text="", style="Muted.TLabel", width=12)
            val.grid(row=r, column=2, sticky="w", padx=8)
            self.bars[key] = (bar, val)

        self.lbl_perf = ttk.Label(tab, text="", style="Muted.TLabel")
        self.lbl_perf.pack(anchor="w", pady=(16, 0))
        self.lbl_backend = ttk.Label(tab, text="", style="Muted.TLabel")
        self.lbl_backend.pack(anchor="w")

    def _build_settings(self, tab) -> None:
        s = self.settings
        canvas_row = 0

        def slider(label, attr, lo, hi, fmt):
            nonlocal canvas_row
            ttk.Label(tab, text=label).grid(row=canvas_row, column=0, sticky="w", pady=3)
            var = tk.DoubleVar(value=getattr(s, attr))
            val = ttk.Label(tab, text=fmt.format(var.get()), style="Muted.TLabel", width=8)

            def on_move(_v, attr=attr, var=var, val=val):
                setattr(s, attr, float(var.get()))
                val.config(text=fmt.format(var.get()))

            ttk.Scale(tab, from_=lo, to=hi, variable=var, command=on_move,
                      length=200).grid(row=canvas_row, column=1, sticky="w", padx=8)
            val.grid(row=canvas_row, column=2, sticky="w")
            canvas_row += 1

        def check(label, attr):
            nonlocal canvas_row
            var = tk.BooleanVar(value=getattr(s, attr))
            ttk.Checkbutton(tab, text=label, variable=var,
                            command=lambda: setattr(s, attr, bool(var.get()))
                            ).grid(row=canvas_row, column=0, columnspan=3, sticky="w", pady=2)
            canvas_row += 1

        ttk.Label(tab, text="Smoothing filter").grid(row=canvas_row, column=0, sticky="w")
        method = tk.StringVar(value=s.smoothing_method)
        cb = ttk.Combobox(tab, textvariable=method, values=["kalman", "ema"],
                          state="readonly", width=10)
        cb.grid(row=canvas_row, column=1, sticky="w", padx=8, pady=3)
        cb.bind("<<ComboboxSelected>>", lambda _e: setattr(s, "smoothing_method", method.get()))
        canvas_row += 1

        slider("Smoothing strength", "smoothing_strength", 0.0, 1.0, "{:.2f}")
        slider("Dwell time (s)", "dwell_time_s", 0.5, 3.0, "{:.1f} s")
        slider("Dwell radius (px)", "dwell_radius_px", 15, 80, "{:.0f} px")
        slider("Right-blink hold (s)", "blink_hold_s", 0.4, 2.0, "{:.1f} s")
        slider("Eye-closed EAR", "ear_closed_threshold", 0.10, 0.30, "{:.2f}")
        slider("Scroll dead-zone", "pitch_deadzone", 0.015, 0.10, "{:.3f}")
        slider("Scroll speed", "scroll_speed", 0.2, 4.0, "{:.1f}")
        check("Dwell left-click enabled", "dwell_enabled")
        check("Blink right-click enabled", "blink_enabled")
        check("Right-click requires a wink (left eye open)", "blink_require_wink")
        check("Head-pitch scrolling enabled", "scroll_enabled")
        check("Invert scroll direction", "scroll_invert")
        check("Mirror camera preview", "mirror_preview")

        ttk.Button(tab, text="Save settings", style="Accent.TButton",
                   command=self._save_settings).grid(row=canvas_row, column=0,
                                                     sticky="w", pady=(10, 0))

    # ================================================================== #
    # Actions
    # ================================================================== #
    def _on_switch(self, value: bool) -> None:
        self.engine.set_control(value)

    def _save_settings(self) -> None:
        self.settings.save()
        self._banner(f"Settings saved to {C.SETTINGS_FILE}", OK)

    def start_calibration(self) -> None:
        if self._calib_window is not None:
            return
        snap = self.engine.snapshot()
        if not snap.running or snap.error:
            self._banner("Cannot calibrate: camera / tracker not running.", DANGER)
            return
        self.overlay.hide()
        session = self.engine.start_calibration()
        self._calib_window = CalibrationWindow(self.root, session,
                                               on_cancel=self.engine.cancel_calibration)

    def _banner(self, text: str, color: str = MUTED) -> None:
        self.banner.config(text=text, fg=color)

    # ================================================================== #
    # Polling loop
    # ================================================================== #
    def _poll(self) -> None:
        try:
            self._refresh(self.engine.snapshot())
        except Exception:
            log.exception("UI refresh failed")
        self.root.after(self.POLL_MS, self._poll)

    def _refresh(self, snap) -> None:
        s = self.settings

        # ---- One-shot events -------------------------------------------- #
        for ev in snap.events:
            if ev[0] == "calibration_done":
                _, ok, msg = ev
                self._banner(msg, OK if ok else DANGER)
        if self._calib_window is not None and self._calib_window.closed:
            self._calib_window = None

        # ---- Preview / errors ------------------------------------------- #
        if snap.error:
            self.preview.config(image="", text=f"⚠  {snap.error}", fg=DANGER)
            self._photo = None
            if not self.retry_btn.winfo_ismapped():
                self.retry_btn.pack(pady=(10, 0))
        else:
            if self.retry_btn.winfo_ismapped():
                self.retry_btn.pack_forget()
            if snap.preview is not None and id(snap.preview) != self._last_preview_id:
                self._last_preview_id = id(snap.preview)
                img = Image.fromarray(snap.preview)
                if img.size != (PREVIEW_W, PREVIEW_H):
                    img = img.resize((PREVIEW_W, PREVIEW_H), Image.BILINEAR)
                self._photo = ImageTk.PhotoImage(img)
                self.preview.config(image=self._photo, text="")

        # ---- Control state ---------------------------------------------- #
        self.switch.set(snap.control_enabled)
        self.switch_label.config(text="ON" if snap.control_enabled else "OFF",
                                 fg=OK if snap.control_enabled else MUTED)
        if snap.error:
            pill, color = "● CAMERA ERROR", DANGER
        elif snap.calibrating:
            pill, color = "● CALIBRATING", ACCENT
        elif snap.control_enabled:
            pill, color = "● CONTROLLING MOUSE", OK
        else:
            pill, color = "● PAUSED", WARN
        self.status_pill.config(text=pill, fg=color)

        # ---- Status tab ------------------------------------------------- #
        self.dwell_ring.set(snap.dwell_progress)
        self.blink_ring.set(snap.right_hold_progress)
        self.lbl_face.config(text="Face: tracked ✓" if snap.face_detected else "Face: not detected",
                             foreground=OK if snap.face_detected else DANGER)
        self.lbl_calib.config(
            text="Calibration: personalised ✓" if snap.calibrated
            else "Calibration: default — run Calibration Mode for accuracy",
            foreground=TEXT if snap.calibrated else WARN)
        if snap.last_action:
            self.lbl_action.config(text=f"Last action: {snap.last_action}")

        thr = s.ear_closed_threshold
        for key, value in (("rear", snap.right_ear), ("lear", snap.left_ear)):
            bar, lbl = self.bars[key]
            bar.set(value, threshold=thr, color=DANGER if value < thr else OK)
            lbl.config(text=f"{value:.3f}")
        bar, lbl = self.bars["pitch"]
        scroll_txt = {1: "▲ scroll", -1: "▼ scroll"}.get(snap.scroll_direction, "neutral")
        bar.set(snap.pitch_offset, band=s.pitch_deadzone,
                color=ACCENT if snap.scroll_direction else MUTED)
        lbl.config(text=f"{snap.pitch_offset:+.3f} {scroll_txt}")

        self.lbl_perf.config(text=f"{snap.fps:.1f} FPS   ·   dropped frames: {snap.dropped_frames}"
                                  + (f"   ·   cursor: {snap.cursor}" if snap.cursor else ""))
        self.lbl_backend.config(text=f"Tracker: {snap.backend or '—'}")

        if snap.message != self._last_message:      # show each new message once
            self._last_message = snap.message
            if snap.message and not snap.events:
                self._banner(snap.message)

        # ---- Cursor overlay --------------------------------------------- #
        if snap.control_enabled and snap.face_detected and not snap.calibrating:
            self.overlay.update(snap.cursor, snap.dwell_progress,
                                snap.right_hold_progress if s.blink_enabled else 0.0,
                                s.dwell_radius_px)
        else:
            self.overlay.hide()

    # ------------------------------------------------------------------ #
    def on_close(self) -> None:
        self.engine.set_control(False)
        self.hotkeys.stop()
        self.engine.stop()
        self.root.destroy()
