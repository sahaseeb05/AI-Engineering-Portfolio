# AirTouchOS

Control your computer by moving your hand in front of the webcam. AirTouchOS is built with MediaPipe Hand Landmarker, OpenCV, PyAutoGUI and NumPy.

```
 webcam ──► HandTracker ──► GestureEngine ──► OSController ──► mouse / keyboard / volume
 (OpenCV)   (MediaPipe       (pure state        (PyAutoGUI,
            Tasks, VIDEO)     machine)           pycaw)
                  └───────────────┴───────────► VisualOverlay ──► preview window
```

| Module | Class | Responsibility |
|---|---|---|
| `airtouchos/hand_tracker.py` | `HandTracker` | Downloads the model and runs the Hand Landmarker in VIDEO mode. Returns 21 landmarks per frame. |
| `airtouchos/gesture_engine.py` | `GestureEngine` | Turns landmarks into gestures and intents. It uses timing, hysteresis, debounce and latches, and has no side effects. |
| `airtouchos/os_controller.py` | `OSController` | Maps the camera region to the screen, applies EMA smoothing and bounds checks, and sends clicks, drags, Alt+F4 and volume changes. |
| `airtouchos/visual_overlay.py` | `VisualOverlay` | Draws the skeleton, pinch lines, progress rings, volume bar, FPS, status and calibration read-out. |
| `airtouchos/config.py` | `AppConfig` | Holds every threshold. Any of them can be overridden from JSON. |
| `airtouchos/app.py` | `AirTouchOS` | Runs the capture loop and the command line, handles keys and makes sure the app shuts down safely. |

## Gestures

| Gesture | How | Action |
|---|---|---|
| **Move** | Point with your index finger (landmark 8) | The cursor follows, with smoothing |
| **Left click** | Quick pinch of the index tip (8) and thumb tip (4), then release | Click at the point where the pinch started (the cursor freezes while you pinch) |
| **Drag & drop** | Keep the index/thumb pinch held longer than 0.35 s, then move | Mouse down, the cursor follows, and releasing the pinch drops |
| **Right click** | Pinch the middle tip (12) and thumb tip (4) | Right click once each time the fingers touch |
| **Close window** | Full fist (all fingertips near the wrist, landmark 0) held for 0.3 s | `Alt+F4`, then a 1.5 s debounce. It fires once per fist. |
| **Volume** | Thumb, index and pinky out, middle and ring folded ("🤟 without the middle"). Then change the **vertical** gap between the index and thumb tips. | Sets the system volume. A bigger gap means louder. |
| **Pause / resume** | Open palm (all 5 fingers out) held still for 2 s | Freezes or unfreezes all mouse input. A ring shows the progress. |

Keyboard shortcuts (the preview window must have focus): **q / Esc** quits, **p** pauses, **c** toggles the calibration read-out. Ctrl+C in the console also exits cleanly. Any held mouse button is always released on exit.

### Safety design

* `pyautogui.FAILSAFE = False` is set so that the cursor can reach the screen corners. It is replaced by the following checks:
  * The cursor is clamped to the screen, and NaN or infinite targets are rejected.
  * Held mouse buttons are released when the hand is lost for more than 0.25 s, when tracking pauses, when a fist is made, on any OS error and on exit.
  * Alt+F4 is **refused** when the desktop or taskbar has focus (on Windows that would open the *Shut Down* dialog). It is also refused for the AirTouchOS preview and console windows.
* The fist must be held for 0.3 s before anything fires. After that it is latched until you open your hand, and there is a 1.5 s debounce. This stops accidental or repeated window closes.
* The left pinch uses hysteresis: it engages below 0.26 and releases above 0.38. This stops "chattering" clicks.

## Setup (Windows 10/11)

1. **Python 3.9–3.12, 64-bit.** MediaPipe may not publish wheels for newer Python versions yet, so run `python --version` to check.
2. **Create a virtual environment and install:**
   ```powershell
   cd C:\Users\User\Desktop\AirOS
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python.exe` directly.
3. **Camera access:** Go to Settings → Privacy & security → Camera and turn on *Let desktop apps access your camera*.
4. **Run it:**
   ```powershell
   python main.py --calibrate
   ```
   The first run downloads `models/hand_landmarker.task` (~7.5 MB). If you are offline, download it yourself from the URL in `airtouchos/config.py` and save it to that path.
5. **Optional: run the tests.** They use no camera and synthetic hands:
   ```powershell
   python -m unittest discover -s tests -v
   ```

Useful options:

```
--camera 1           use a second webcam
--width 1280 --height 720   higher resolution (more precise, lower FPS)
--scale 0.6          smaller preview window
--no-topmost         don't keep the preview above other windows
--config my.json     load calibrated thresholds
--dump-config my.json  write all current settings to a file, then exit
--debug              verbose logs
```

**pycaw** is optional. With it, the volume gesture sets an *absolute* level from 0 to 100 %. Without it, AirTouchOS falls back to the media keys. In that mode the gap works as a rate control: above 65 % of the range the volume goes up repeatedly, below 35 % it goes down, and in between it holds. Yellow lines on the volume bar mark these zones.

## Threshold calibration guide

Every hand distance is measured in **palm lengths**, meaning the distance from the wrist (0) to the middle-finger MCP (9). This makes the thresholds independent of how far you sit from the camera. Calibrate once at your usual distance and lighting.

### Step 0: Environment
* Sit about 50–80 cm from the camera, with your hand at chest height.
* Use even, front-facing light. Avoid a bright window behind you.
* Check that the FPS counter is green (≥ 20). If it is red, lower the resolution (`--width 640 --height 480`) and close other apps that use the camera.

### Step 1: Write out the config
```powershell
python main.py --dump-config my.json
```
Edit `my.json`, then run `python main.py --config my.json --calibrate`. Remove any keys you are not changing if you want a short file. Only the keys you include override the defaults.

### Step 2: Screen reach (`margin_x`, `margin_y`)
The grey **"screen area"** rectangle maps to your whole screen.
* Move your index tip to each corner of the rectangle. The cursor should reach each screen corner without your hand leaving the frame.
* If reaching the corners is hard, **increase** the margins (0.20 / 0.22). This makes the rectangle smaller, so the cursor moves faster.
* If the cursor feels too fast or twitchy, **decrease** them (0.10 / 0.12).

### Step 3: Smoothness (`ema_alpha_min`, `ema_alpha_max`, `cursor_deadzone_px`)
* The cursor shivers when your hand is still → lower `ema_alpha_min` (0.12) or raise `cursor_deadzone_px` (2.5).
* The cursor lags behind fast moves → raise `ema_alpha_max` (0.7).
* The cursor drifts when you pinch to click → lower `precision_alpha_factor` (0.2) or raise `precision_zone_factor` (2.2).

### Step 4: Pinch thresholds (`left_pinch_on/off`, `right_pinch_on/off`)
With calibration on, the panel shows `L pinch` and `R pinch` live.
1. Pinch index and thumb **firmly** and note the value (typically 0.08–0.20). Call it **P**.
2. Relax your hand into the pointing pose and note the value (typically 0.6–1.2). Call it **R**.
3. Set `left_pinch_on ≈ P + 0.3·(R−P)` and `left_pinch_off ≈ on + 0.10`.
4. Repeat for the middle finger and thumb to get the `right_pinch_*` values.
* Clicks fire when you didn't mean them to → lower `*_on`.
* Pinches don't register → raise `*_on`, and keep `*_off` at least 0.08 above it.
* Drags start when you meant to click → raise `drag_hold_s` (0.5).

### Step 5: Fist (`fist_tip_ratio`, `fist_thumb_ratio`)
The panel lists each finger's **tip-wrist** ratio. A value turns **red** when it passes the fist test.
1. Make a relaxed but full fist. Note the **largest** of the index, middle, ring and pinky values (typically 0.7–1.0), and note the thumb value.
2. Make a pointing hand and a thumbs-up. Note the lowest finger value in those poses that is not part of the fist.
3. Set `fist_tip_ratio` about 0.1 above your fist maximum. Keep it well below the ratio of any extended finger, which is usually above 1.5.
4. Set `fist_thumb_ratio` between your fist thumb value and your thumbs-up thumb value.
* Accidental closes → lower the ratios, or raise `fist_confirm_s` to 0.5.
* The fist is not detected → raise the ratios by 0.05 at a time.
* `fist_debounce_s` (1.5) is the minimum time between two Alt+F4s.

### Step 6: Volume (`volume_span_min`, `volume_span_max`)
Make the volume pose. The panel shows `vol span`.
1. Bring the index and thumb tips to the same height and note the span. Set `volume_span_min` a little above it.
2. Spread them as far apart vertically as is comfortable and note the span. Set `volume_span_max` a little below it.
* The volume jumps around → lower `volume_smoothing` (0.2) or raise `volume_step_pct` (5).

### Step 7: Pause (`palm_hold_s`, `palm_still_radius`)
* Pausing triggers while you are just moving around → raise `palm_hold_s` (3.0) or lower `palm_still_radius` (0.025).
* It is hard to hold still long enough → raise `palm_still_radius` (0.05).

### Step 8: Finger up/down detection (`finger_extended_ratio`, `thumb_extended_ratio`)
The `EXT/fold` labels in the panel should match your fingers.
* A bent finger still reads `EXT` → raise `finger_extended_ratio` (1.2).
* A straight finger reads `fold` → lower it (1.02).
* Adjust the thumb the same way with `thumb_extended_ratio`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open camera index 0` | Another app is using the webcam, or camera privacy is off. Try `--camera 1`. |
| The cursor moves in the opposite horizontal direction | Don't use `--no-mirror`. It is meant for cameras that are already mirrored. |
| Cursor off by a factor on a HiDPI display | PyAutoGUI makes the process DPI-aware on Windows. If it is still wrong, set Python's *Properties → Compatibility → Change high DPI settings → Override → Application*. |
| Alt+F4 "does nothing" | The focused window is protected (see the banner). Click the window you want to close first. |
| Gestures stop in admin windows (Task Manager, UAC) | Windows blocks input from normal-privilege processes into elevated windows. Run the terminal as administrator if you need this. |
| Low FPS | Use 640×480, close other camera apps, and plug in your laptop (power-saving throttles the CPU). |
