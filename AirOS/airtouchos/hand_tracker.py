"""
HandTracker: webcam frame -> 21 hand landmarks via MediaPipe Hand Landmarker.

Uses the MediaPipe *Tasks* API (``mediapipe.tasks.python.vision.HandLandmarker``)
in VIDEO running mode, which enables temporal tracking between frames and is
considerably more stable than running per-image detection.  The ``.task``
model bundle is downloaded automatically on first run.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from .config import AppConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe hand topology)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Bone connections used for drawing the skeleton (same as mp.solutions.hands).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm edge
)


@dataclass(frozen=True)
class HandObservation:
    """One detected hand in one frame.

    Attributes:
        landmarks_norm: (21, 3) float array, x/y in [0, 1] relative to the frame
            (already in the mirrored frame if mirroring is enabled), z relative
            depth with the wrist as origin.
        landmarks_px: (21, 2) float array of pixel coordinates in the frame.
        handedness: "Left" or "Right" as reported by MediaPipe.
        score: handedness confidence in [0, 1].
        frame_size: (width, height) of the frame the landmarks refer to.
        timestamp: ``time.perf_counter()`` value at capture.
    """

    landmarks_norm: np.ndarray
    landmarks_px: np.ndarray
    handedness: str
    score: float
    frame_size: tuple[int, int]
    timestamp: float


class HandTracker:
    """Wraps the MediaPipe HandLandmarker in VIDEO mode for a single hand."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        model_path = Path(config.model_path)
        if not model_path.is_absolute():
            # Resolve relative paths against the project root, not the CWD.
            model_path = Path(__file__).resolve().parent.parent / model_path
        model_bytes = self._load_model_bytes(model_path, config.model_url)
        options = mp_vision.HandLandmarkerOptions(
            # Passing the model as a buffer avoids MediaPipe's known problems
            # with non-ASCII characters in Windows file paths.
            base_options=mp_tasks.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        log.info("HandLandmarker ready (VIDEO mode, 1 hand).")

    # ------------------------------------------------------------------ model
    @staticmethod
    def _load_model_bytes(path: Path, url: str) -> bytes:
        """Return the model bundle, downloading it to *path* if missing."""
        if not path.is_file() or path.stat().st_size == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading hand landmarker model from %s", url)
            tmp_path = path.with_suffix(path.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=60) as response, open(tmp_path, "wb") as fh:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        fh.write(chunk)
                tmp_path.replace(path)
            except OSError as exc:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download the model to {path}. Download it manually from\n"
                    f"  {url}\nand place it at that path."
                ) from exc
            log.info("Model saved to %s (%d KB)", path, path.stat().st_size // 1024)
        return path.read_bytes()

    # -------------------------------------------------------------- inference
    def process(self, frame_bgr: np.ndarray) -> Optional[HandObservation]:
        """Detect/track a hand in a BGR frame (already mirrored if desired)."""
        captured_at = time.perf_counter()
        height, width = frame_bgr.shape[:2]

        # VIDEO mode requires strictly increasing timestamps in milliseconds.
        timestamp_ms = int(captured_at * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        points = result.hand_landmarks[0]
        norm = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float64)
        px = norm[:, :2] * np.array([width, height], dtype=np.float64)

        handedness, score = "Unknown", 0.0
        if result.handedness and result.handedness[0]:
            category = result.handedness[0][0]
            handedness, score = category.category_name, float(category.score)

        return HandObservation(
            landmarks_norm=norm,
            landmarks_px=px,
            handedness=handedness,
            score=score,
            frame_size=(width, height),
            timestamp=captured_at,
        )

    # -------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Release the native MediaPipe graph."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
