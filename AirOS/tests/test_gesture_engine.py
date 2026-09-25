"""
Deterministic tests for GestureEngine using synthetic hand landmarks.

Run with:  python -m unittest discover -s tests -v
No camera, model or OS input is used.
"""

import unittest

import numpy as np

from airtouchos.config import AppConfig
from airtouchos.gesture_engine import Gesture, GestureEngine, compute_metrics
from airtouchos.hand_tracker import HandObservation

FRAME = (640, 480)
DT = 1 / 30  # simulated frame period

# Palm geometry in pixels (palm length wrist->middle MCP = 100 px).
WRIST = (320.0, 400.0)
MCPS = {"index": (290.0, 310.0), "middle": (320.0, 300.0),
        "ring": (350.0, 305.0), "pinky": (375.0, 320.0)}
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(extended=("thumb", "index", "middle", "ring", "pinky"),
              thumb_tip=None, dx=0.0, dy=0.0) -> HandObservation:
    """Build a 21-landmark hand.  Fingers not in *extended* are curled into
    the palm.  *thumb_tip* overrides the thumb tip position (for pinches)."""
    px = np.zeros((21, 2))
    px[0] = WRIST
    # Thumb chain: CMC, MCP, IP, TIP.
    if "thumb" in extended:
        px[1:5] = [(295, 385), (265, 360), (245, 340), (230, 320)]
    else:
        px[1:5] = [(300, 385), (285, 365), (275, 345), (310, 340)]
    for name, base in FINGER_BASE.items():
        mx, my = MCPS[name]
        px[base] = (mx, my)
        if name in extended:
            px[base + 1:base + 4] = [(mx, my - 40), (mx, my - 70), (mx, my - 95)]
        else:
            px[base + 1:base + 4] = [(mx, my - 30), (mx, my - 10), (mx, my + 30)]
    if thumb_tip is not None:
        px[4] = thumb_tip
    px += (dx, dy)
    w, h = FRAME
    norm = np.zeros((21, 3))
    norm[:, 0] = px[:, 0] / w
    norm[:, 1] = px[:, 1] / h
    return HandObservation(norm, px, "Right", 0.99, FRAME, 0.0)


POINTER = make_hand(extended=("thumb", "index"))
OPEN_PALM = make_hand()
FIST = make_hand(extended=())
INDEX_TIP = tuple(POINTER.landmarks_px[8])
MIDDLE_TIP_OPEN = tuple(OPEN_PALM.landmarks_px[12])
LEFT_PINCH = make_hand(extended=("thumb", "index"), thumb_tip=(INDEX_TIP[0] - 5, INDEX_TIP[1] + 8))
RIGHT_PINCH = make_hand(extended=("thumb", "middle"),
                        thumb_tip=(MIDDLE_TIP_OPEN[0] - 5, MIDDLE_TIP_OPEN[1] + 8))
VOLUME_HIGH = make_hand(extended=("thumb", "index", "pinky"))


class EngineHarness:
    """Feeds frames at a fixed rate and collects outputs."""

    def __init__(self, cfg=None):
        self.cfg = cfg or AppConfig()
        self.engine = GestureEngine(self.cfg)
        self.t = 100.0

    def feed(self, obs, seconds=DT):
        outs = []
        for _ in range(max(1, round(seconds / DT))):
            self.t += DT
            outs.append(self.engine.update(obs, self.t))
        return outs


def count(outs, attr):
    return sum(1 for o in outs if getattr(o, attr))


class MetricsTest(unittest.TestCase):
    def test_pose_classification(self):
        cfg = AppConfig()
        self.assertEqual(compute_metrics(OPEN_PALM, cfg).extended, (True,) * 5)
        self.assertEqual(compute_metrics(FIST, cfg).extended, (False,) * 5)
        self.assertEqual(compute_metrics(POINTER, cfg).extended, (True, True, False, False, False))
        m = compute_metrics(LEFT_PINCH, cfg)
        self.assertLess(m.left_pinch, cfg.left_pinch_on)
        self.assertAlmostEqual(m.scale_px, 100.0)


class GestureEngineTest(unittest.TestCase):
    def test_pointer_moves_cursor(self):
        h = EngineHarness()
        out = h.feed(POINTER)[-1]
        self.assertEqual(out.gesture, Gesture.POINTER)
        self.assertAlmostEqual(out.cursor_target[0], INDEX_TIP[0] / FRAME[0])
        self.assertAlmostEqual(out.cursor_target[1], INDEX_TIP[1] / FRAME[1])

    def test_quick_pinch_is_left_click(self):
        h = EngineHarness()
        h.feed(POINTER, 0.2)
        during = h.feed(LEFT_PINCH, 0.15)
        self.assertTrue(all(o.cursor_target is None for o in during), "cursor frozen while pinched")
        after = h.feed(POINTER, 0.1)
        self.assertEqual(count(during + after, "left_click"), 1)
        self.assertEqual(count(during + after, "drag_start"), 0)

    def test_held_pinch_drags_and_drops(self):
        h = EngineHarness()
        h.feed(POINTER, 0.2)
        held = h.feed(LEFT_PINCH, 0.8)
        self.assertEqual(count(held, "drag_start"), 1)
        self.assertEqual(held[-1].gesture, Gesture.DRAG)
        self.assertIsNotNone(held[-1].cursor_target, "cursor follows during drag")
        released = h.feed(POINTER, 0.1)
        self.assertEqual(count(released, "drag_end"), 1)
        self.assertEqual(count(held + released, "left_click"), 0)

    def test_right_pinch_fires_once_per_contact(self):
        h = EngineHarness()
        h.feed(POINTER, 0.2)
        held = h.feed(RIGHT_PINCH, 1.0)
        self.assertEqual(count(held, "right_click"), 1)
        self.assertEqual(count(held, "left_click"), 0)
        h.feed(POINTER, 0.5)
        self.assertEqual(count(h.feed(RIGHT_PINCH, 0.2), "right_click"), 1)

    def test_fist_closes_once_with_debounce(self):
        h = EngineHarness()
        h.feed(POINTER, 0.2)
        first = h.feed(FIST, 1.0)
        self.assertEqual(count(first, "close_window"), 1, "latched while fist held")
        h.feed(POINTER, 0.2)
        second = h.feed(FIST, 0.5)            # re-fist inside the 1.5 s window
        self.assertEqual(count(second, "close_window"), 0)
        h.feed(POINTER, 1.0)
        third = h.feed(FIST, 0.5)             # well after the debounce window
        self.assertEqual(count(third, "close_window"), 1)

    def test_brief_fist_does_not_close(self):
        h = EngineHarness()
        h.feed(POINTER, 0.2)
        outs = h.feed(FIST, 0.15) + h.feed(POINTER, 0.2)
        self.assertEqual(count(outs, "close_window"), 0)

    def test_open_palm_hold_toggles_pause(self):
        h = EngineHarness()
        outs = h.feed(OPEN_PALM, 2.2)
        self.assertEqual(count(outs, "pause_toggled"), 1)
        self.assertTrue(h.engine.paused)
        paused = h.feed(POINTER, 0.3) + h.feed(LEFT_PINCH, 0.3) + h.feed(FIST, 1.0)
        self.assertTrue(all(o.cursor_target is None for o in paused))
        self.assertEqual(count(paused, "left_click") + count(paused, "close_window"), 0)
        h.feed(POINTER, 0.2)
        self.assertEqual(count(h.feed(OPEN_PALM, 2.2), "pause_toggled"), 1)
        self.assertFalse(h.engine.paused)

    def test_moving_palm_does_not_pause(self):
        h = EngineHarness()
        outs = []
        for i in range(90):  # 3 s of a palm sweeping sideways
            outs += h.feed(make_hand(dx=(i % 30) * 4.0))
        self.assertEqual(count(outs, "pause_toggled"), 0)

    def test_volume_mode(self):
        h = EngineHarness()
        outs = h.feed(VOLUME_HIGH, 1.0)
        self.assertTrue(all(o.gesture == Gesture.VOLUME for o in outs))
        levels = [o.volume_level for o in outs if o.volume_level is not None]
        self.assertTrue(levels, "volume emitted after confirm time")
        expected = np.interp(compute_metrics(VOLUME_HIGH, h.cfg).volume_span,
                             [h.cfg.volume_span_min, h.cfg.volume_span_max], [0, 1])
        self.assertAlmostEqual(levels[-1], expected, delta=0.03)
        self.assertEqual(count(outs, "left_click"), 0)

    def test_hand_loss_during_drag_releases_button(self):
        h = EngineHarness()
        h.feed(LEFT_PINCH, 0.6)
        self.assertTrue(h.engine.dragging)
        blip = h.feed(None, DT)               # single dropped frame: keep dragging
        self.assertEqual(count(blip, "drag_end"), 0)
        lost = h.feed(None, 0.5)
        self.assertEqual(count(lost, "drag_end"), 1)
        self.assertFalse(h.engine.dragging)


if __name__ == "__main__":
    unittest.main()
