"""
Cursor motion smoothing.

Raw iris estimates jitter by several pixels every frame (sensor noise,
landmark noise and physiological micro-saccades). Two filters are provided;
both operate on *normalised* screen coordinates (0..1) so their tuning is
independent of monitor resolution.

* ``EMAFilter``     – Exponential Moving Average. Trivial and robust, but lag
                      grows as smoothing increases.
* ``KalmanFilter2D``– Constant-velocity Kalman filter. Predicts motion, so for
                      the same steadiness it lags noticeably less than EMA
                      during deliberate gaze shifts. Default.

``smoothing_strength`` (0..1) from the settings maps onto each filter's
parameters so the user only needs one slider.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Point = Tuple[float, float]


class EMAFilter:
    """y_t = alpha * x_t + (1 - alpha) * y_{t-1}"""

    def __init__(self, strength: float = 0.6):
        self.alpha = 0.5
        self.set_strength(strength)
        self._state: Optional[np.ndarray] = None

    def set_strength(self, strength: float) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        # strength 0 -> alpha 0.65 (light), strength 1 -> alpha 0.05 (heavy)
        self.alpha = 0.65 - 0.60 * strength

    def reset(self) -> None:
        self._state = None

    def update(self, point: Point, dt: float = 1 / 30) -> Point:
        x = np.asarray(point, dtype=np.float64)
        if self._state is None:
            self._state = x
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return float(self._state[0]), float(self._state[1])


class KalmanFilter2D:
    """
    Constant-velocity Kalman filter with state [x, y, vx, vy].

    Process noise follows the discrete white-noise-acceleration model, so the
    filter is well behaved at any (variable) frame rate.
    """

    ACCEL_STD = 3.0   # expected gaze acceleration, screen-widths / s^2

    def __init__(self, strength: float = 0.6):
        self.meas_std = 0.03
        self.set_strength(strength)
        self.reset()

    def set_strength(self, strength: float) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        # Larger measurement noise => filter trusts its prediction more.
        self.meas_std = 0.004 + 0.07 * strength

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._P = np.eye(4)

    def update(self, point: Point, dt: float = 1 / 30) -> Point:
        z = np.asarray(point, dtype=np.float64)
        dt = float(np.clip(dt, 1e-3, 0.2))

        if self._x is None:
            self._x = np.array([z[0], z[1], 0.0, 0.0])
            self._P = np.diag([self.meas_std ** 2] * 2 + [1.0, 1.0])
            return float(z[0]), float(z[1])

        # ---- Predict ------------------------------------------------------ #
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)
        g = np.array([0.5 * dt * dt, dt])
        q1 = np.outer(g, g) * self.ACCEL_STD ** 2       # 1-axis block
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q1
        Q[np.ix_([1, 3], [1, 3])] = q1

        x = F @ self._x
        P = F @ self._P @ F.T + Q

        # ---- Update ------------------------------------------------------- #
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float64)
        R = np.eye(2) * self.meas_std ** 2
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        self._x = x + K @ y
        self._P = (np.eye(4) - K @ H) @ P
        return float(self._x[0]), float(self._x[1])


def make_filter(method: str, strength: float):
    return EMAFilter(strength) if method.lower() == "ema" else KalmanFilter2D(strength)
