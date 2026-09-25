"""Generate synthetic health telemetry with clinical U-shaped risk targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = ["age", "blood_pressure", "cholesterol", "blood_sugar", "heart_rate"]
BASELINES = {"age": 30.0, "blood_pressure": 120.0, "cholesterol": 170.0, "blood_sugar": 90.0, "heart_rate": 72.0}
DEVIATION_SCALES = {"age": 70.0, "blood_pressure": 80.0, "cholesterol": 190.0, "blood_sugar": 210.0, "heart_rate": 78.0}


def _u_distance(values: np.ndarray, baseline: float, scale: float) -> np.ndarray:
    return np.clip(np.abs(values - baseline) / scale, 0, 1)


def generate_dataset(rows: int = 1600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    telemetry = {"age": rng.uniform(18, 100, rows), "blood_pressure": rng.uniform(80, 200, rows), "cholesterol": rng.uniform(100, 360, rows), "blood_sugar": rng.uniform(55, 300, rows), "heart_rate": rng.uniform(40, 150, rows)}
    distances = {name: _u_distance(telemetry[name], BASELINES[name], DEVIATION_SCALES[name]) for name in FEATURE_NAMES}
    age, blood_pressure = telemetry["age"], telemetry["blood_pressure"]
    cholesterol, blood_sugar = telemetry["cholesterol"], telemetry["blood_sugar"]
    severe_low = 0.34 * np.clip((90 - blood_pressure) / 30, 0, 1) ** 0.65 + 0.30 * np.clip((70 - blood_sugar) / 25, 0, 1) ** 0.65 + 0.18 * np.clip((120 - cholesterol) / 40, 0, 1) ** 0.65 + 0.10 * np.clip((55 - age) / 37, 0, 1) ** 0.65
    severe_high = 0.34 * np.clip((blood_pressure - 160) / 40, 0, 1) ** 0.65 + 0.30 * np.clip((blood_sugar - 180) / 120, 0, 1) ** 0.65 + 0.18 * np.clip((cholesterol - 240) / 120, 0, 1) ** 0.65 + 0.10 * np.clip((age - 65) / 35, 0, 1) ** 0.65
    deviation_risk = sum(weight * distances[name] ** 1.35 for name, weight in zip(FEATURE_NAMES, (0.18, 0.27, 0.20, 0.23, 0.12)))
    critical_signal = np.clip(severe_low + severe_high, 0, 1)
    risk_score = 100 * np.clip(0.60 * deviation_risk + 0.70 * critical_signal + 0.18 * critical_signal ** 0.65, 0, 1) + rng.normal(0, 1.8, rows)
    return pd.DataFrame({name: np.round(telemetry[name], 1) for name in FEATURE_NAMES} | {"risk_score": np.round(np.clip(risk_score, 0, 100), 2)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "data" / "raw_dataset.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Generated {args.rows:,} records at {args.output}")


if __name__ == "__main__":
    main()
