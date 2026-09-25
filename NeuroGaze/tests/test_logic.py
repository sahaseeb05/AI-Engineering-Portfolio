"""
Hardware-free tests for the pure logic (run: python -m pytest tests  or
python tests/test_logic.py). No webcam, screen or mouse is touched.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegisgaze.calibration import GazeMapper  # noqa: E402
from aegisgaze.face_tracker import eye_aspect_ratio, iris_in_eye  # noqa: E402
from aegisgaze.gestures import BlinkDetector, DwellClicker, HeadScroller  # noqa: E402
from aegisgaze.smoothing import EMAFilter, KalmanFilter2D  # noqa: E402


def test_dwell_fires_once_after_hold():
    d = DwellClicker(radius=30, dwell_s=1.2, cooldown_s=1.0)
    clicks = [d.update((500 + (i % 3), 400), i * 0.05)[1] for i in range(60)]  # 3 s
    assert sum(clicks) == 1
    assert clicks.index(True) * 0.05 >= 1.2


def test_dwell_resets_when_cursor_moves():
    d = DwellClicker(radius=30, dwell_s=1.2, cooldown_s=1.0)
    t, fired = 0.0, False
    for i in range(40):  # jump 50 px every 0.5 s -> never dwells
        t += 0.05
        fired |= d.update((100 + 50 * (i // 10), 100), t)[1]
    assert not fired


def test_blink_requires_hold_and_fires_once():
    b = BlinkDetector(threshold=0.2, hold_s=0.8, require_wink=False, cooldown_s=1.0)
    # natural 0.3 s blink -> nothing
    assert not any(b.update(0.1, 0.1, t * 0.05).fired for t in range(6))
    b.update(0.3, 0.3, 1.0)
    fires = [b.update(0.1, 0.3, 2.0 + t * 0.05).fired for t in range(40)]
    assert sum(fires) == 1


def test_wink_mode_ignores_both_eyes_closed():
    b = BlinkDetector(threshold=0.2, hold_s=0.8, require_wink=True, cooldown_s=1.0)
    assert not any(b.update(0.1, 0.1, t * 0.05).fired for t in range(40))


def test_scroll_direction_and_deadzone():
    s = HeadScroller(deadzone=0.04, speed=1.0, interval_s=0.1)
    assert s.update(0.02, 1.0) == 0.0
    assert s.update(-0.08, 2.0) > 0      # head up -> scroll up
    assert s.update(+0.08, 3.0) < 0      # head down -> scroll down


def test_filters_reduce_jitter_and_converge():
    rng = np.random.default_rng(0)
    for f in (EMAFilter(0.6), KalmanFilter2D(0.6)):
        out = [f.update((0.5 + rng.normal(0, 0.02), 0.5 + rng.normal(0, 0.02)), 1 / 30)
               for _ in range(200)]
        out = np.array(out[50:])
        assert out.std(axis=0).max() < 0.012
        assert np.allclose(out.mean(axis=0), 0.5, atol=0.01)


def test_homography_calibration_roundtrip():
    m = GazeMapper(1920, 1080)
    gaze = [(0.50, -0.02), (0.56, -0.06), (0.44, -0.06), (0.44, 0.02), (0.56, 0.02)]
    screen = [(960, 540), (60, 60), (1860, 60), (1860, 1020), (60, 1020)]
    ok, _ = m.fit(gaze, screen)
    assert ok
    for g, s in zip(gaze[1:], screen[1:]):
        assert np.allclose(m.map(g), s, atol=2)


def test_calibration_rejects_no_eye_movement():
    m = GazeMapper(1920, 1080)
    ok, _ = m.fit([(0.5, 0.0)] * 5, [(960, 540), (60, 60), (1860, 60), (1860, 1020), (60, 1020)])
    assert not ok and not m.calibrated


def test_geometry_helpers():
    pts = np.zeros((478, 2))
    # open eye: width 30, lid gap 10 -> EAR = 10/30
    idx = [0, 1, 2, 3, 4, 5]
    pts[0], pts[3] = (0, 0), (30, 0)
    pts[1], pts[5] = (10, -5), (10, 5)
    pts[2], pts[4] = (20, -5), (20, 5)
    assert abs(eye_aspect_ratio(pts, idx) - 10 / 30) < 1e-9
    gx, gy = iris_in_eye(pts, np.array([15.0, 3.0]), (0, 3))
    assert abs(gx - 0.5) < 1e-9 and abs(gy - 0.1) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
