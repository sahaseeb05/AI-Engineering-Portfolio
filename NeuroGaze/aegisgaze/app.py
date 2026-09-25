"""Application bootstrap: logging, dependency checks, wiring, main loop."""

from __future__ import annotations

import importlib
import logging
import sys
import tkinter as tk
from tkinter import messagebox

from . import config as C

REQUIRED = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "pyautogui": "pyautogui",
    "numpy": "numpy",
    "PIL": "pillow",
}


def _setup_logging() -> None:
    C.APP_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(C.APP_DIR / "aegisgaze.log", encoding="utf-8")],
    )


def _check_dependencies() -> list[str]:
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    return missing


def main() -> int:
    _setup_logging()
    log = logging.getLogger("aegisgaze")

    missing = _check_dependencies()
    if missing:
        msg = ("Missing Python packages: " + ", ".join(missing) +
               "\n\nInstall them with:\n    pip install -r requirements.txt")
        log.error(msg)
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("AegisGaze – missing dependencies", msg)
        except tk.TclError:
            pass
        return 1

    # Import PyAutoGUI (via MouseController) BEFORE creating any Tk window:
    # on Windows it makes the process DPI-aware, so Tk, the calibration
    # targets and the injected mouse events all share the same pixel grid.
    from .engine import Engine
    from .mouse import MouseController
    from .ui.dashboard import Dashboard

    settings = C.Settings.load()
    mouse = MouseController()
    log.info("Screen resolution: %dx%d", mouse.screen_w, mouse.screen_h)
    engine = Engine(settings, mouse)

    root = tk.Tk()

    def report_tk_error(exc_type, exc, tb):
        # Any unexpected UI exception: make sure we are not left controlling
        # the mouse with a broken interface.
        mouse.set_enabled(False)
        log.error("Unhandled UI error", exc_info=(exc_type, exc, tb))

    root.report_callback_exception = report_tk_error

    Dashboard(root, engine, settings)
    engine.start()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        mouse.set_enabled(False)
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
