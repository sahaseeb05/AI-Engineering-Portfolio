"""
Processing engine: runs in a background thread and owns the whole pipeline

    camera -> landmarks -> gaze mapping -> smoothing -> gestures -> mouse

The Tk GUI never touches OpenCV/MediaPipe directly; it polls
``Engine.snapshot()`` (a copy of the latest state) and sends commands via
thread-safe methods. This keeps the UI responsive even if inference stalls.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui

from . import config as C
from .calibration import CalibrationSession, GazeMapper
from .camera import Camera, CameraError
from .face_tracker import FaceData, FaceTracker
from .gestures import BlinkDetector, DwellClicker, HeadScroller
from .mouse import MouseController
from .smoothing import make_filter

log = logging.getLogger(__name__)


@dataclass
class EngineState:
    """Immutable-by-convention snapshot shared with the GUI."""

    running: bool = False
    control_enabled: bool = False
    preview: Optional[np.ndarray] = None        # annotated RGB frame
    fps: float = 0.0
    dropped_frames: int = 0
    face_detected: bool = False
    backend: str = ""
    cursor: Optional[Tuple[int, int]] = None
    dwell_progress: float = 0.0
    right_hold_progress: float = 0.0
    right_ear: float = 0.0
    left_ear: float = 0.0
    pitch_offset: float = 0.0
    scroll_direction: int = 0                   # +1 up, -1 down, 0 none
    calibrated: bool = False
    calibrating: bool = False
    last_action: str = ""
    message: str = ""                           # info banner for the user
    error: str = ""                             # fatal/camera errors
    events: list = field(default_factory=list)  # one-shot events for the UI


class Engine:
    NEUTRAL_WARMUP_FRAMES = 20   # frames used to auto-estimate neutral pitch

    def __init__(self, settings: C.Settings, mouse: MouseController):
        self.settings = settings
        self.mouse = mouse
        self.mapper = GazeMapper(mouse.screen_w, mouse.screen_h)
        if self.mapper.load():
            log.info("Loaded saved calibration.")

        self._state = EngineState(calibrated=self.mapper.calibrated)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._calibration: Optional[CalibrationSession] = None
        self._pending_camera: Optional[int] = None
        self._recenter_requested = False
        self._pitch_warmup: list[float] = []

        s = settings
        self._filter = make_filter(s.smoothing_method, s.smoothing_strength)
        self._filter_method = s.smoothing_method
        self._dwell = DwellClicker(s.dwell_radius_px, s.dwell_time_s, s.dwell_cooldown_s)
        self._blink = BlinkDetector(s.ear_closed_threshold, s.blink_hold_s,
                                    s.blink_require_wink, s.blink_cooldown_s)
        self._scroller = HeadScroller(s.pitch_deadzone, s.scroll_speed,
                                      s.scroll_interval_s, s.scroll_invert)
        self._cursor: Optional[Tuple[float, float]] = None
        self._frame_times: deque[float] = deque(maxlen=30)

    # ================================================================== #
    # Public, thread-safe API used by the GUI / hotkeys
    # ================================================================== #
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="AegisEngine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.mouse.set_enabled(False)
        if self._thread is not None:
            self._thread.join(timeout=3)

    def snapshot(self) -> EngineState:
        """Return a copy of the latest state and clear one-shot events."""
        with self._lock:
            snap = replace(self._state, events=list(self._state.events))
            self._state.events.clear()
        snap.control_enabled = self.mouse.enabled
        return snap

    def set_control(self, enabled: bool) -> None:
        if enabled and self._calibration is not None:
            return  # never inject input while calibrating
        self.mouse.set_enabled(enabled)
        if not enabled:
            self._dwell.reset()

    def panic(self) -> None:
        """Immediate, unconditional stop (Esc)."""
        self.mouse.set_enabled(False)
        self._dwell.reset()
        self._post(last_action="PANIC STOP (Esc)")

    def toggle_control(self) -> None:
        self.set_control(not self.mouse.enabled)

    def start_calibration(self) -> CalibrationSession:
        self.mouse.set_enabled(False)
        session = CalibrationSession(self.mouse.screen_w, self.mouse.screen_h, self.settings)
        self._calibration = session
        return session

    def cancel_calibration(self) -> None:
        if self._calibration is not None:
            self._calibration.cancel()
            self._calibration = None
            self._post(message="Calibration cancelled.")

    def recenter_head(self) -> None:
        """Use the current head pose as the neutral (no-scroll) pitch."""
        self._recenter_requested = True

    def reconnect_camera(self, index: int) -> None:
        self._pending_camera = index

    # ================================================================== #
    # Worker thread
    # ================================================================== #
    def _post(self, **changes) -> None:
        with self._lock:
            events = changes.pop("event", None)
            for key, value in changes.items():
                setattr(self._state, key, value)
            if events:
                self._state.events.append(events)

    def _sync_settings(self) -> None:
        """Push live settings (edited in the GUI) into the components."""
        s = self.settings
        if s.smoothing_method != self._filter_method:
            self._filter = make_filter(s.smoothing_method, s.smoothing_strength)
            self._filter_method = s.smoothing_method
        self._filter.set_strength(s.smoothing_strength)
        self._dwell.radius, self._dwell.dwell_s = s.dwell_radius_px, s.dwell_time_s
        self._dwell.cooldown_s = s.dwell_cooldown_s
        self._blink.threshold, self._blink.hold_s = s.ear_closed_threshold, s.blink_hold_s
        self._blink.require_wink = s.blink_require_wink
        self._blink.cooldown_s = s.blink_cooldown_s
        self._scroller.deadzone, self._scroller.speed = s.pitch_deadzone, s.scroll_speed
        self._scroller.interval_s, self._scroller.invert = s.scroll_interval_s, s.scroll_invert

    def _run(self) -> None:
        try:
            tracker = FaceTracker()
        except Exception as exc:
            log.exception("Landmark model failed to load")
            self._post(error=f"Face tracker failed to start:\n{exc}")
            return

        camera = Camera(self.settings.camera_index,
                        self.settings.frame_width, self.settings.frame_height)
        self._post(running=True, backend=tracker.backend_name)
        last_t = time.monotonic()

        try:
            while not self._stop.is_set():
                # ---- Camera (re)connection ------------------------------- #
                if self._pending_camera is not None:
                    camera.release()
                    camera.index = self.settings.camera_index = self._pending_camera
                    self._pending_camera = None
                try:
                    frame = camera.read()
                except CameraError as exc:
                    self.mouse.set_enabled(False)      # fail safe
                    self._post(error=str(exc), face_detected=False, preview=None)
                    self._stop.wait(2.0)               # retry periodically
                    continue
                if frame is None:                      # dropped frame
                    self._post(dropped_frames=camera.dropped_frames)
                    continue
                if self._state.error:
                    self._post(error="", message="Camera connected.")

                now = time.monotonic()
                dt, last_t = now - last_t, now
                self._frame_times.append(now)
                self._sync_settings()

                face = tracker.process(frame)
                try:
                    self._process(face, now, dt)
                except pyautogui.FailSafeException:
                    self.mouse.set_enabled(False)
                    self._post(message="Fail-safe: mouse was pushed into a screen "
                                       "corner. Control disabled.")

                preview = self._annotate(frame, face)
                fps = 0.0
                if len(self._frame_times) > 1:
                    span = self._frame_times[-1] - self._frame_times[0]
                    fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
                self._post(preview=preview, fps=fps, dropped_frames=camera.dropped_frames)
        except Exception as exc:  # last-resort guard: never leave control on
            log.exception("Engine crashed")
            self.mouse.set_enabled(False)
            self._post(error=f"Processing error: {exc}")
        finally:
            camera.release()
            tracker.close()
            self._post(running=False)

    # ------------------------------------------------------------------ #
    def _process(self, face: Optional[FaceData], now: float, dt: float) -> None:
        s = self.settings
        enabled = self.mouse.enabled

        # ---- Calibration mode ------------------------------------------- #
        session = self._calibration
        if session is not None:
            session.feed(face)
            if session.finished:
                ok, msg = session.apply(self.mapper)
                self._calibration = None
                self._filter.reset()
                self._post(calibrated=self.mapper.calibrated, calibrating=False,
                           message=msg, event=("calibration_done", ok, msg))
            else:
                self._post(calibrating=True)
        else:
            self._post(calibrating=False)

        if face is None:
            self._dwell.reset()
            self._post(face_detected=False, dwell_progress=0.0,
                       right_hold_progress=0.0, scroll_direction=0)
            return

        # ---- Neutral head pitch ----------------------------------------- #
        if self._recenter_requested:
            self._recenter_requested = False
            self.mapper.neutral_pitch = face.pitch
            self.mapper.save()
            self._post(message="Head position re-centred.")
        if self.mapper.neutral_pitch is None:
            self._pitch_warmup.append(face.pitch)
            if len(self._pitch_warmup) >= self.NEUTRAL_WARMUP_FRAMES:
                self.mapper.neutral_pitch = float(np.median(self._pitch_warmup))
        neutral = self.mapper.neutral_pitch if self.mapper.neutral_pitch is not None else face.pitch
        pitch_offset = face.pitch - neutral

        # ---- Blink / right click ---------------------------------------- #
        blink = self._blink.update(face.right_ear, face.left_ear, now)
        eyes_closed = blink.right_closed or blink.left_closed
        if blink.fired and enabled and session is None and s.blink_enabled:
            self.mouse.right_click()
            self._post(last_action="Right click (blink)")

        # ---- Head pitch / scroll ---------------------------------------- #
        scrolling = s.scroll_enabled and self._scroller.is_active(pitch_offset)
        scroll_dir = 0
        if scrolling and session is None:
            notches = self._scroller.update(pitch_offset, now)
            scroll_dir = 1 if pitch_offset < 0 else -1
            if s.scroll_invert:
                scroll_dir = -scroll_dir
            if notches and enabled:
                self.mouse.scroll(notches)
                self._post(last_action=f"Scroll {'up' if notches > 0 else 'down'}")

        # ---- Gaze -> cursor --------------------------------------------- #
        # Closed eyes give garbage iris positions: freeze the cursor instead.
        if not eyes_closed and session is None:
            sx, sy = self.mapper.map(face.gaze)
            nx, ny = self._filter.update((sx / self.mouse.screen_w,
                                          sy / self.mouse.screen_h), dt)
            target = self.mouse.clamp(nx * self.mouse.screen_w, ny * self.mouse.screen_h)
            if self._cursor is None or math.dist(target, self._cursor) >= s.cursor_deadzone_px:
                self._cursor = target
                self.mouse.move(*target)

        # ---- Dwell / left click ----------------------------------------- #
        dwell_progress = 0.0
        if not (enabled and s.dwell_enabled) or scrolling or session is not None:
            self._dwell.reset()
        elif not eyes_closed and self._cursor is not None:
            # (While blinking we simply skip updating: a natural blink must
            #  neither reset nor advance an ongoing dwell.)
            dwell_progress, clicked = self._dwell.update(self._cursor, now)
            if clicked:
                self.mouse.left_click()
                self._post(last_action="Left click (dwell)")

        self._post(face_detected=True,
                   cursor=self._cursor,
                   dwell_progress=dwell_progress,
                   right_hold_progress=blink.hold_progress,
                   right_ear=face.right_ear, left_ear=face.left_ear,
                   pitch_offset=pitch_offset,
                   scroll_direction=scroll_dir)

    # ------------------------------------------------------------------ #
    def _annotate(self, frame: np.ndarray, face: Optional[FaceData]) -> np.ndarray:
        """Draw mesh, eye contours and iris circles; return an RGB image."""
        mirror = self.settings.mirror_preview
        img = cv2.flip(frame, 1) if mirror else frame.copy()
        w = img.shape[1]

        if face is not None:
            pts = face.landmarks.copy()
            if mirror:
                pts[:, 0] = w - 1 - pts[:, 0]
            ipts = pts.astype(np.int32)

            # Light face mesh (every landmark as a dot).
            for x, y in ipts[:468]:
                cv2.circle(img, (int(x), int(y)), 1, (120, 200, 120), -1, cv2.LINE_AA)

            closed_color, open_color = (60, 60, 255), (255, 200, 0)
            for contour, ear in ((C.RIGHT_EYE_CONTOUR, face.right_ear),
                                 (C.LEFT_EYE_CONTOUR, face.left_ear)):
                color = closed_color if ear < self.settings.ear_closed_threshold else open_color
                cv2.polylines(img, [ipts[contour]], True, color, 1, cv2.LINE_AA)

            for ring in (C.RIGHT_IRIS, C.LEFT_IRIS):
                centre = pts[ring[0]]
                radius = float(np.mean([np.linalg.norm(pts[i] - centre) for i in ring[1:]]))
                c = (int(centre[0]), int(centre[1]))
                cv2.circle(img, c, max(2, int(radius)), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(img, c, 2, (0, 0, 255), -1, cv2.LINE_AA)

            # Head-pitch axis: forehead -> nose -> chin.
            for a, b in ((C.FOREHEAD, C.NOSE_TIP), (C.NOSE_TIP, C.CHIN)):
                cv2.line(img, tuple(ipts[a]), tuple(ipts[b]), (255, 120, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (60, 60, 255), 2, cv2.LINE_AA)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
