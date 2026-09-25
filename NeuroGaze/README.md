# AegisGaze — AI Eye-Gaze & Head-Pose Laptop Controller

Hands-free mouse control using only a standard webcam, for users with limited
or no hand mobility (ALS, spinal-cord injury, paralysis).

| Action        | How                                                                 |
|---------------|----------------------------------------------------------------------|
| Move cursor   | Look where you want the cursor to go                                 |
| Left click    | Keep your gaze within 30 px for 1.2 s (a blue ring fills around the cursor) |
| Right click   | Close your **right** eye for 0.8 s (an orange ring fills)            |
| Scroll up/down| Tilt your head slightly up / down                                    |
| **Panic stop**| **Esc**, from any app. It only ever *disables* control.              |
| Toggle        | **F8**, or the switch in the dashboard                               |

## Install & run

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py                    # or: python -m aegisgaze
```

Requires Python 3.9 – 3.12 and a webcam. On first launch with a recent MediaPipe
release, the face model (~4 MB) is downloaded once to `~/.aegisgaze/`.

## First-time setup

1. Sit at your normal distance with your face evenly lit.
2. Click **Calibration Mode**. Keep your head still and follow the dot with
   your eyes: centre, then the four corners (about 15 s total). Press Esc to cancel.
3. Turn on **Mouse control**.
4. If scrolling starts on its own, sit in your normal position and click
   **Re-centre head**.

Calibration and settings are saved in `~/.aegisgaze/` and reloaded on start.
Recalibrate if you change position, lighting or screen resolution.

## How it works

```
webcam ─► MediaPipe 478-pt mesh ─► iris-in-eye gaze feature ─► homography (calibration)
        (incl. iris 468-477)      EAR per eye, head-pitch index    ─► Kalman / EMA smoothing
                                                                   ─► dwell / blink / scroll ─► PyAutoGUI
```

* **Gaze feature.** The iris centre is measured in an eye-local frame: along
  the corner-to-corner line and perpendicular to it, normalised by eye width.
  This follows where the eye looks rather than where the head is, and it is
  unaffected by head roll and distance.
* **Mapping.** A least-squares homography fitted to the 5 calibration points
  also handles camera offset and tilt. Points outside the calibrated area
  extrapolate, so the cursor can reach the full screen.
* **Smoothing.** A constant-velocity Kalman filter is the default. It lags less
  than EMA for the same steadiness. There is also a small pixel dead-zone.
  Both filters are switchable in Settings.
* **Safety.** All input goes through one enable flag, and Esc clears it
  instantly. Control switches off on camera loss, on errors, during
  calibration, and if the PyAutoGUI corner fail-safe triggers. The cursor
  freezes while the eyes are closed.

## Project layout

```
aegisgaze/
  app.py              bootstrap, dependency check, logging
  config.py           all thresholds + landmark indices (Settings dataclass)
  camera.py           webcam with backend fallback, dropped-frame + reconnect handling
  face_tracker.py     MediaPipe (legacy Face Mesh or Tasks FaceLandmarker), EAR, gaze, pitch
  smoothing.py        EMA and 2-D Kalman filters
  calibration.py      5-point calibration session + homography GazeMapper
  gestures.py         DwellClicker, BlinkDetector, HeadScroller (pure logic)
  mouse.py            thread-safe PyAutoGUI wrapper (the single enable flag)
  hotkeys.py          global Esc / F8 via pynput
  engine.py           background processing thread
  ui/dashboard.py     Tkinter dashboard
  ui/overlay.py       click-through dwell ring that follows the cursor (Windows)
  ui/calibration_window.py
tests/test_logic.py   hardware-free tests (python tests/test_logic.py)
```

## Tuning tips

* **Right clicks you didn't intend:** raise *Right-blink hold*, lower *Eye-closed EAR*,
  or enable *requires a wink*.
* **Right eye never registers as closed:** raise *Eye-closed EAR*. Watch the live
  EAR bar to find your value.
* **Cursor shaky:** raise *Smoothing strength*. **Cursor laggy:** lower it.
* **Accidental scrolling:** raise *Scroll dead-zone*.

## Limitations

A webcam-only tracker is less precise than dedicated IR eye trackers. Expect
accuracy in the range of a few centimetres on screen. Vertical gaze is
harder to measure than horizontal gaze. Good, even front lighting and a
fresh calibration help the most. The on-screen dwell ring is shown on
Windows only (it needs a click-through window). On other platforms the
progress rings appear in the dashboard.
