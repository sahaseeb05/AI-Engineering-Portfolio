"""
Gaze -> screen calibration.

The user looks at 5 targets (screen centre + 4 corners). For each target we
record the median gaze feature; a least-squares **homography** then maps any
gaze feature to screen pixels. A homography (rather than simple min/max
scaling) absorbs camera tilt, the camera sitting above/below the screen and
the slight "keystone" shape of eye rotation. Points outside the calibrated
quad extrapolate naturally, so the cursor can reach 100 % of the screen even
though the targets are inset from the edges.

The centre target also captures the neutral head pitch used for scrolling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .face_tracker import FaceData

log = logging.getLogger(__name__)


class GazeMapper:
    """Maps the normalised iris-in-eye feature to screen pixel coordinates."""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.H: Optional[np.ndarray] = None
        self.neutral_pitch: Optional[float] = None
        self.calibrated = False
        self._set_default()

    # ------------------------------------------------------------------ #
    def _set_default(self) -> None:
        """
        Generic mapping for an average adult until calibration is run.
        In the raw (un-mirrored) camera image the iris moves towards the
        image's LEFT when the user looks to THEIR right, hence the x-flip.
        """
        src = np.float32([[0.56, -0.06], [0.44, -0.06], [0.44, 0.02], [0.56, 0.02]])
        dst = np.float32([[0, 0], [self.screen_w, 0],
                          [self.screen_w, self.screen_h], [0, self.screen_h]])
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.calibrated = False

    def map(self, gaze: Tuple[float, float]) -> Tuple[float, float]:
        """Return (x, y) in screen pixels (may lie outside the screen)."""
        p = np.array([[gaze]], dtype=np.float64)
        x, y = cv2.perspectiveTransform(p, self.H)[0, 0]
        return float(x), float(y)

    # ------------------------------------------------------------------ #
    def fit(self, gaze_pts: List[Tuple[float, float]],
            screen_pts: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """Fit a homography; returns (success, human readable message)."""
        src = np.asarray(gaze_pts, dtype=np.float64)
        dst = np.asarray(screen_pts, dtype=np.float64)

        # Sanity check: did the eyes actually move between targets?
        spread = src.max(axis=0) - src.min(axis=0)
        if spread[0] < 0.02 or spread[1] < 0.008:
            return False, ("Eye movement between targets was too small. Keep "
                           "your head still and move only your eyes, then retry.")

        H, _ = cv2.findHomography(src, dst, 0)
        if H is None or not np.all(np.isfinite(H)):
            return False, "Calibration failed (degenerate points). Please retry."

        # Validate: reprojection error on the calibration points.
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
        diag = float(np.hypot(self.screen_w, self.screen_h))
        if err > 0.25 * diag:
            return False, (f"Calibration inconsistent (mean error {err:.0f}px). "
                           "Please retry and keep your head still.")

        self.H = H
        self.calibrated = True
        return True, f"Calibration successful (mean fit error {err:.0f}px)."

    # ------------------------------------------------------------------ #
    def save(self) -> None:
        try:
            C.APP_DIR.mkdir(parents=True, exist_ok=True)
            C.CALIBRATION_FILE.write_text(json.dumps({
                "screen": [self.screen_w, self.screen_h],
                "H": self.H.tolist(),
                "neutral_pitch": self.neutral_pitch,
            }, indent=2), encoding="utf-8")
        except OSError as exc:
            log.error("Could not save calibration: %s", exc)

    def load(self) -> bool:
        try:
            data = json.loads(C.CALIBRATION_FILE.read_text(encoding="utf-8"))
            if list(data["screen"]) != [self.screen_w, self.screen_h]:
                log.info("Screen resolution changed; recalibration required.")
                return False
            H = np.array(data["H"], dtype=np.float64)
            if H.shape != (3, 3) or not np.all(np.isfinite(H)):
                return False
            self.H = H
            self.neutral_pitch = data.get("neutral_pitch")
            self.calibrated = True
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            log.warning("Ignoring invalid calibration file: %s", exc)
            return False


# --------------------------------------------------------------------------- #
@dataclass
class CalibrationTarget:
    name: str
    screen_xy: Tuple[int, int]
    gaze_samples: List[Tuple[float, float]] = field(default_factory=list)
    pitch_samples: List[float] = field(default_factory=list)


class CalibrationSession:
    """
    Time-driven state machine fed by the processing thread.

    For each target: ``settle`` seconds for the eyes to arrive (ignored),
    then ``sample`` seconds of collection. Frames with closed eyes are
    discarded, since blinking corrupts the iris estimate.
    """

    MIN_SAMPLES = 8

    def __init__(self, screen_w: int, screen_h: int, settings: C.Settings):
        m = settings.calib_target_margin_px
        self.settle_s = settings.calib_settle_s
        self.sample_s = settings.calib_sample_s
        self.ear_threshold = settings.ear_closed_threshold
        self.targets = [
            CalibrationTarget("centre", (screen_w // 2, screen_h // 2)),
            CalibrationTarget("top-left", (m, m)),
            CalibrationTarget("top-right", (screen_w - m, m)),
            CalibrationTarget("bottom-right", (screen_w - m, screen_h - m)),
            CalibrationTarget("bottom-left", (m, screen_h - m)),
        ]
        self.index = 0
        self._target_start = time.monotonic()
        self.finished = False
        self.cancelled = False

    # -- State queried by the UI (thread-safe: simple reads) -------------- #
    @property
    def current(self) -> Optional[CalibrationTarget]:
        return self.targets[self.index] if self.index < len(self.targets) else None

    def phase(self) -> Tuple[str, float]:
        """('settle'|'sample'|'done', progress 0..1 within the phase)."""
        if self.finished or self.current is None:
            return "done", 1.0
        t = time.monotonic() - self._target_start
        if t < self.settle_s:
            return "settle", t / self.settle_s
        return "sample", min(1.0, (t - self.settle_s) / self.sample_s)

    # -- Fed by processing thread ---------------------------------------- #
    def feed(self, face: Optional[FaceData]) -> None:
        if self.finished or self.cancelled:
            return
        target = self.current
        phase, progress = self.phase()

        if phase == "sample" and face is not None:
            eyes_open = min(face.right_ear, face.left_ear) > self.ear_threshold
            if eyes_open:
                target.gaze_samples.append(face.gaze)
                target.pitch_samples.append(face.pitch)

        if phase == "sample" and progress >= 1.0:
            if len(target.gaze_samples) < self.MIN_SAMPLES:
                # Not enough good frames (face lost / eyes closed): extend.
                self._target_start = time.monotonic() - self.settle_s
                return
            self.index += 1
            self._target_start = time.monotonic()
            if self.index >= len(self.targets):
                self.finished = True

    def cancel(self) -> None:
        self.cancelled = True

    def apply(self, mapper: GazeMapper) -> Tuple[bool, str]:
        """Fit the mapper from the collected samples."""
        gaze = [tuple(np.median(np.array(t.gaze_samples), axis=0)) for t in self.targets]
        screen = [t.screen_xy for t in self.targets]
        ok, msg = mapper.fit(gaze, screen)
        if ok:
            mapper.neutral_pitch = float(np.median(self.targets[0].pitch_samples))
            mapper.save()
        return ok, msg
