"""
Robust webcam wrapper.

Handles the failure modes that matter for an assistive device that must run
unattended for hours:
  * camera missing / busy at start-up  -> ``CameraError`` with a clear message
  * transient dropped frames            -> tolerated, counted
  * camera unplugged / driver crash     -> automatic reconnect with back-off
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or has permanently failed."""


class Camera:
    # Consecutive failed reads before we assume the device is gone.
    MAX_CONSECUTIVE_FAILURES = 30
    RECONNECT_DELAY_S = 1.0

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._failures = 0
        self.dropped_frames = 0

    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """Open the camera, trying the most reliable backend per platform."""
        self.release()
        backends = [cv2.CAP_ANY]
        if platform.system() == "Windows":
            # DirectShow opens much faster than MSMF on most Windows laptops.
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif platform.system() == "Darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always get the newest frame
                ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    self._failures = 0
                    log.info("Camera %d opened (backend %d).", self.index, backend)
                    return
            if cap is not None:
                cap.release()

        raise CameraError(
            f"Could not open webcam #{self.index}. Check that it is connected, "
            "not used by another application (Zoom, Teams, browser), and that "
            "camera access is allowed in the OS privacy settings."
        )

    def read(self) -> Optional[np.ndarray]:
        """
        Return the next BGR frame, or ``None`` if this frame was dropped.

        Raises ``CameraError`` only when reconnection is impossible.
        """
        if self._cap is None:
            self.open()

        ok, frame = self._cap.read()
        if ok and frame is not None and frame.size > 0:
            self._failures = 0
            return frame

        # --- Frame dropped ------------------------------------------------ #
        self._failures += 1
        self.dropped_frames += 1
        if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
            log.warning("Camera stopped delivering frames; reconnecting...")
            time.sleep(self.RECONNECT_DELAY_S)
            self.open()  # raises CameraError if the device is really gone
        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
