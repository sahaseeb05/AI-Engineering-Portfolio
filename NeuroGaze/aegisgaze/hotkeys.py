"""
System-wide keyboard hotkeys (work even when the dashboard is not focused).

* Panic key (default ``Esc``) – ALWAYS disables mouse control immediately.
  It never re-enables, so pressing Esc in another program can only ever make
  the system safer.
* Toggle key (default ``F8``) – switches control on/off, e.g. for a caregiver.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


class HotkeyListener:
    def __init__(self, panic_key: str, toggle_key: str,
                 on_panic: Callable[[], None], on_toggle: Callable[[], None]):
        self._panic_name = panic_key.lower()
        self._toggle_name = toggle_key.lower()
        self._on_panic = on_panic
        self._on_toggle = on_toggle
        self._listener = None

    def start(self) -> bool:
        """Start the global listener; returns False if unavailable."""
        try:
            from pynput import keyboard
        except Exception as exc:  # missing package / no display server
            log.warning("Global hotkeys unavailable (%s). Esc works only "
                        "while the dashboard window is focused.", exc)
            return False

        def key_name(key) -> str:
            if isinstance(key, keyboard.Key):
                return key.name.lower()
            return (getattr(key, "char", "") or "").lower()

        def on_press(key):
            try:
                name = key_name(key)
                if name == self._panic_name:
                    self._on_panic()
                elif name == self._toggle_name:
                    self._on_toggle()
            except Exception:  # never let a callback kill the hook thread
                log.exception("Hotkey handler failed")

        self._listener = keyboard.Listener(on_press=on_press)  # daemon thread
        self._listener.start()
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
