"""
AirTouchOS application: wires HandTracker -> GestureEngine -> OSController
-> VisualOverlay together in a single real-time capture loop, plus the CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from typing import Optional

import cv2

from .config import AppConfig
from .gesture_engine import GestureEngine
from .hand_tracker import HandTracker
from .os_controller import OSController
from .visual_overlay import VisualOverlay

log = logging.getLogger("airtouchos")

_KEY_ESC = 27
_MAX_READ_FAILURES = 60   # ~2 s of failed reads at 30 FPS before giving up


class AirTouchOS:
    """Top-level application object owning the camera and all subsystems."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.tracker = HandTracker(config)
        self.engine = GestureEngine(config)
        self.controller = OSController(config)
        self.overlay = VisualOverlay(config)
        self.cap: Optional[cv2.VideoCapture] = None

    # ----------------------------------------------------------------- camera
    def _open_camera(self) -> cv2.VideoCapture:
        """Open the webcam, preferring DirectShow on Windows for low latency."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(self.cfg.camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # MJPG lets most USB webcams deliver 30 FPS at 640x480+.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always process the newest frame
            ok, _ = cap.read()
            if ok:
                log.info("Camera %d opened: %dx%d @ %.0f FPS (backend %s)",
                         self.cfg.camera_index,
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         cap.get(cv2.CAP_PROP_FPS), cap.getBackendName())
                return cap
            cap.release()
        raise RuntimeError(
            f"Could not open camera index {self.cfg.camera_index}. "
            "Try --camera 1, close other apps using the webcam, or check "
            "Windows Settings > Privacy & security > Camera."
        )

    def _setup_window(self) -> None:
        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_AUTOSIZE)
        if self.cfg.window_topmost:
            try:
                cv2.setWindowProperty(self.cfg.window_name, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                log.debug("WND_PROP_TOPMOST not supported by this OpenCV build.")

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        """Main capture/recognise/act/draw loop.  Returns on quit."""
        read_failures = 0
        had_hand = False
        try:
            self.cap = self._open_camera()
            self._setup_window()
            log.info("Running. Focus the preview window and press q or Esc to quit.")
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures > _MAX_READ_FAILURES:
                        raise RuntimeError("Camera stopped delivering frames.")
                    continue
                read_failures = 0

                if self.cfg.mirror:
                    # Mirror BEFORE inference so landmarks, overlay and cursor
                    # direction all agree with what the user sees.
                    frame = cv2.flip(frame, 1)

                obs = self.tracker.process(frame)
                now = time.perf_counter()

                if obs is not None and not had_hand:
                    self.controller.reset_smoothing()
                had_hand = obs is not None

                out = self.engine.update(obs, now)
                try:
                    self.controller.apply(out)
                except Exception:
                    # An OS-side failure must never leave a button held down.
                    log.exception("OS action failed; releasing input.")
                    self.controller.release_all()
                if out.pause_toggled:
                    self.controller.release_all()
                    self.controller.reset_smoothing()

                current_volume = (
                    self.controller.volume.current_level()
                    if out.volume_level is not None else None
                )
                self.overlay.draw(frame, obs, out, self.controller.volume.mode,
                                  current_volume, self.controller.last_blocked_reason)

                if self.cfg.preview_scale != 1.0:
                    frame = cv2.resize(frame, None, fx=self.cfg.preview_scale,
                                       fy=self.cfg.preview_scale,
                                       interpolation=cv2.INTER_LINEAR)
                cv2.imshow(self.cfg.window_name, frame)

                if not self._handle_keys(cv2.waitKey(1) & 0xFF):
                    break
                if cv2.getWindowProperty(self.cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("Preview window closed.")
                    break
        finally:
            self.shutdown()

    def _handle_keys(self, key: int) -> bool:
        """Process a key press from the preview window; False means quit."""
        if key in (ord("q"), ord("Q"), _KEY_ESC):
            return False
        if key in (ord("p"), ord("P")):
            paused = not self.engine.paused
            self.engine.set_paused(paused)
            self.controller.release_all()
            self.controller.reset_smoothing()
            self.overlay.flash("TRACKING PAUSED" if paused else "TRACKING RESUMED")
        elif key in (ord("c"), ord("C")):
            self.overlay.show_calibration = not self.overlay.show_calibration
        return True

    def shutdown(self) -> None:
        """Release every resource.  Safe to call more than once."""
        self.controller.release_all()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.tracker.close()
        cv2.destroyAllWindows()
        log.info("AirTouchOS stopped.")


# ---------------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airtouchos",
        description="Touchless hand-gesture OS controller (MediaPipe + OpenCV + PyAutoGUI).",
    )
    parser.add_argument("--config", help="JSON file overriding any AppConfig field")
    parser.add_argument("--dump-config", metavar="PATH",
                        help="write the effective configuration to PATH and exit")
    parser.add_argument("--camera", type=int, help="webcam index (default 0)")
    parser.add_argument("--width", type=int, help="capture width in px")
    parser.add_argument("--height", type=int, help="capture height in px")
    parser.add_argument("--scale", type=float, help="preview window scale factor")
    parser.add_argument("--calibrate", action="store_true",
                        help="start with the live calibration read-out visible")
    parser.add_argument("--no-mirror", action="store_true", help="disable horizontal flip")
    parser.add_argument("--no-topmost", action="store_true",
                        help="do not keep the preview window above other windows")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()
    if args.config:
        cfg = AppConfig.from_json(args.config, cfg)
    overrides = {}
    if args.camera is not None:
        overrides["camera_index"] = args.camera
    if args.width is not None:
        overrides["frame_width"] = args.width
    if args.height is not None:
        overrides["frame_height"] = args.height
    if args.scale is not None:
        overrides["preview_scale"] = args.scale
    if args.calibrate:
        overrides["show_calibration"] = True
    if args.no_mirror:
        overrides["mirror"] = False
    if args.no_topmost:
        overrides["window_topmost"] = False
    return replace(cfg, **overrides).validated()


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cfg = config_from_args(args)
    except (ValueError, OSError) as exc:
        log.error("%s", exc)
        return 2

    if args.dump_config:
        cfg.to_json(args.dump_config)
        log.info("Configuration written to %s", args.dump_config)
        return 0

    try:
        AirTouchOS(cfg).run()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    return 0
