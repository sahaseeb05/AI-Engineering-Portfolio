"""Train and evaluate the Fuzzy Medical AI model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from models.fuzzy_engine import FEATURE_NAMES
from models.neuro_fuzzy_model import NeuroFuzzyRegressor

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "raw_dataset.csv"
MODEL_PATH = ROOT / "models" / "saved_model.pkl"


def train_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH, seed: int = 42) -> dict[str, float]:
    frame = pd.read_csv(data_path)
    records = frame[FEATURE_NAMES].to_dict(orient="records")
    targets = frame["risk_score"].to_numpy()
    train_records, test_records, train_targets, test_targets = train_test_split(records, targets, test_size=0.2, random_state=seed)
    model = NeuroFuzzyRegressor(random_state=seed).fit(train_records, train_targets)
    predictions = model.predict(test_records)
    metrics = {"rmse": round(float(mean_squared_error(test_targets, predictions) ** 0.5), 3), "r2": round(float(r2_score(test_targets, predictions)), 4), "samples": float(len(frame))}
    artifact = {"model": model, "metrics": metrics, "features": FEATURE_NAMES, "schema_version": 1}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    temporary_path.replace(model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    metrics = train_model(args.data, args.model)
    print(f"RMSE: {metrics['rmse']} | R2: {metrics['r2']} | Samples: {int(metrics['samples'])}")


if __name__ == "__main__":
    main()
