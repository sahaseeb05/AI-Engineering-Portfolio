"""Merge verified feedback and replace the model only when validation improves."""

from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from train import MODEL_PATH, DATA_PATH, train_model  # noqa: E402

DB_PATH = Path(__file__).parents[1] / "data" / "user_feedback.db"


def _verified_feedback() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            "SELECT age, blood_pressure, cholesterol, blood_sugar, heart_rate, risk_score "
            "FROM feedback WHERE verified = 1 AND risk_score IS NOT NULL "
            "ORDER BY created_at ASC, id ASC",
            connection,
        )


def retrain() -> dict:
    feedback = _verified_feedback()
    if feedback.empty:
        return {"status": "skipped", "reason": "No verified feedback records found."}

    base = pd.read_csv(DATA_PATH)
    merged = pd.concat([base, feedback], ignore_index=True).drop_duplicates(subset=["age", "blood_pressure", "cholesterol", "blood_sugar", "heart_rate", "risk_score"])
    merged_path = DATA_PATH.with_suffix(".merged.csv")
    candidate_path = MODEL_PATH.with_suffix(".candidate.pkl")
    merged.to_csv(merged_path, index=False)
    try:
        candidate_metrics = train_model(merged_path, candidate_path)
        current_metrics = {"r2": -1.0, "rmse": float("inf")}
        if MODEL_PATH.exists():
            with MODEL_PATH.open("rb") as handle:
                current_metrics = pickle.load(handle).get("metrics", current_metrics)
        improved = candidate_metrics["r2"] >= current_metrics.get("r2", -1.0) or candidate_metrics["rmse"] <= current_metrics.get("rmse", float("inf"))
        if improved:
            candidate_path.replace(MODEL_PATH)
            merged_path.replace(DATA_PATH)
            return {"status": "updated", "previous": current_metrics, "candidate": candidate_metrics, "feedback_records": len(feedback)}
        return {"status": "kept_current", "previous": current_metrics, "candidate": candidate_metrics, "feedback_records": len(feedback)}
    finally:
        if merged_path.exists():
            merged_path.unlink()
        if candidate_path.exists():
            candidate_path.unlink()


if __name__ == "__main__":
    print(retrain())
