"""
Facial landmark + iris tracking built on MediaPipe.

Two interchangeable backends produce the same 478-point Face Mesh topology
(468 face points + 10 iris points, i.e. ``refine_landmarks=True``):

* **Legacy Face Mesh** – ``mp.solutions.face_mesh`` (MediaPipe <= 0.10.x).
* **Tasks FaceLandmarker** – the modern API. Recent MediaPipe releases
  (>= 0.10.30 / 1.x) removed ``mp.solutions`` entirely, so this backend is
  used automatically there. Its model file is downloaded once on first run.

From the landmarks we derive the three signals the controller needs:
  1. a head-motion-tolerant *gaze feature* (iris position inside each eye),
  2. the Eye Aspect Ratio (EAR) of each eye, for blink detection,
  3. a *pitch index* describing how far the head is tilted up / down.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config as C

log = logging.getLogger(__name__)


@dataclass
class FaceData:
    """Per-frame measurements for one detected face (pixel coordinates)."""

    landmarks: np.ndarray            # (478, 2) float, raw (un-mirrored) frame px
    gaze: Tuple[float, float]        # normalised iris-in-eye position
    right_ear: float
    left_ear: float
    pitch: float                     # nose position along forehead->chin axis
    right_iris: Tuple[np.ndarray, float]   # (centre, radius)
    left_iris: Tuple[np.ndarray, float]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def eye_aspect_ratio(pts: np.ndarray, idx: list[int]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). ~0.3 open, <0.2 closed."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


def iris_in_eye(pts: np.ndarray, iris_centre: np.ndarray,
                corners: Tuple[int, int]) -> Tuple[float, float]:
    """
    Express the iris centre in an eye-local coordinate frame.

    The x-axis runs along the line joining the two eye corners and the y-axis
    is perpendicular to it; both are normalised by eye width. Because the
    frame moves with the eye, the result mostly reflects *where the eye looks*
    rather than where the head is, and it is invariant to head roll & distance.
    """
    a, b = pts[corners[0]], pts[corners[1]]
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return 0.5, 0.0
    u = axis / width                      # along the eye
    n = np.array([-u[1], u[0]])           # perpendicular, pointing "down" in image
    d = iris_centre - a
    return float(d @ u) / width, float(d @ n) / width


def head_pitch_index(pts: np.ndarray) -> float:
    """
    Position of the nose tip along the forehead->chin axis (0..1).

    Tilting the head up moves the protruding nose tip towards the forehead in
    the 2-D projection (value decreases); tilting down increases it. The value
    is compared against a per-user neutral baseline, so the absolute number
    does not matter.
    """
    top, nose, chin = pts[C.FOREHEAD], pts[C.NOSE_TIP], pts[C.CHIN]
    axis = chin - top
    denom = float(axis @ axis)
    if denom < 1e-6:
        return 0.5
    return float((nose - top) @ axis) / denom


def _iris_circle(pts: np.ndarray, ring: list[int]) -> Tuple[np.ndarray, float]:
    centre = pts[ring[0]]
    radius = float(np.mean([np.linalg.norm(pts[i] - centre) for i in ring[1:]]))
    return centre, radius


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _LegacyFaceMesh:
    name = "MediaPipe Face Mesh (legacy)"

    def __init__(self):
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # adds iris landmarks 468-477
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, rgb: np.ndarray):
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def close(self):
        self._mesh.close()


class _TasksFaceLandmarker:
    name = "MediaPipe FaceLandmarker (Tasks)"

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        _ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(C.MODEL_FILE)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray):
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self):
        self._landmarker.close()


def _ensure_model() -> None:
    """Download the FaceLandmarker model on first use."""
    if C.MODEL_FILE.exists() and C.MODEL_FILE.stat().st_size > 0:
        return
    C.APP_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading face landmark model to %s ...", C.MODEL_FILE)
    tmp = C.MODEL_FILE.with_suffix(".part")
    try:
        with urllib.request.urlopen(C.MODEL_URL, timeout=30) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
        tmp.replace(C.MODEL_FILE)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe face model. Connect to the "
            f"internet once, or place it manually at {C.MODEL_FILE}.\n"
            f"URL: {C.MODEL_URL}\nReason: {exc}"
        ) from exc


def _create_backend():
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return _LegacyFaceMesh()
    return _TasksFaceLandmarker()


# --------------------------------------------------------------------------- #
# Public tracker
# --------------------------------------------------------------------------- #
class FaceTracker:
    def __init__(self):
        self._backend = _create_backend()
        log.info("Landmark backend: %s", self._backend.name)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceData]:
        """Run landmark detection on a raw BGR frame; ``None`` if no face."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # lets MediaPipe avoid a copy
        landmarks = self._backend.detect(rgb)
        if landmarks is None or len(landmarks) < 478:
            return None

        # Normalised -> pixel coordinates. Doing geometry in pixels keeps the
        # x and y axes on the same scale (normalised coords are anisotropic).
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float64)

        r_iris = _iris_circle(pts, C.RIGHT_IRIS)
        l_iris = _iris_circle(pts, C.LEFT_IRIS)
        rx, ry = iris_in_eye(pts, r_iris[0], C.RIGHT_EYE_CORNERS)
        lx, ly = iris_in_eye(pts, l_iris[0], C.LEFT_EYE_CORNERS)

        return FaceData(
            landmarks=pts,
            gaze=((rx + lx) / 2.0, (ry + ly) / 2.0),
            right_ear=eye_aspect_ratio(pts, C.RIGHT_EYE_EAR),
            left_ear=eye_aspect_ratio(pts, C.LEFT_EYE_EAR),
            pitch=head_pitch_index(pts),
            right_iris=r_iris,
            left_iris=l_iris,
        )

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception:
            pass
