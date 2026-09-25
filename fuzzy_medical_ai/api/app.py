"""FastAPI service for prediction, feedback capture, and the static workbench."""

from __future__ import annotations

import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from models.fuzzy_engine import FEATURE_LABELS, FEATURE_NAMES, FEATURE_UNITS, FuzzyEngine

ROOT = Path(__file__).parents[1]
DATA_PATH = ROOT / "data" / "raw_dataset.csv"
DB_PATH = ROOT / "data" / "user_feedback.db"
MODEL_PATH = ROOT / "models" / "saved_model.pkl"


class PatientTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: Annotated[float, Field(ge=18, le=100)]
    blood_pressure: Annotated[float, Field(ge=80, le=200)]
    cholesterol: Annotated[float, Field(ge=100, le=360)]
    blood_sugar: Annotated[float, Field(ge=55, le=300)]
    heart_rate: Annotated[float, Field(ge=40, le=150)]


class FeedbackRecord(PatientTelemetry):
    risk_score: Annotated[float, Field(ge=0, le=100)]
    verified: bool = True
    source: str = Field(default="verified_doctor", max_length=40)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL NOT NULL, blood_pressure REAL NOT NULL, cholesterol REAL NOT NULL,
            blood_sugar REAL NOT NULL, heart_rate REAL NOT NULL, risk_score REAL NOT NULL,
            verified INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL, created_at TEXT NOT NULL
        )
    """)
    connection.commit()
    return connection


def _load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is not available. Run train.py first.")
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def _factors(telemetry: dict[str, float], artifact: dict, fuzzy_result: dict) -> list[dict]:
    model = artifact["model"]
    importance = model.feature_importances()
    factor_rows = []
    for feature in FEATURE_NAMES:
        membership = fuzzy_result["memberships"][feature]
        severity = max(
            membership["Critically_Low"],
            membership["Critically_High"],
            membership["High_Normal"] * 0.65,
            membership["Low_Normal"] * 0.65,
        )
        contribution = severity * (
            importance.get(feature, 0)
            + importance.get(f"{feature}_critically_low", 0)
            + importance.get(f"{feature}_critically_high", 0)
            + importance.get(f"{feature}_high_normal", 0)
            + importance.get(f"{feature}_low_normal", 0)
        )
        factor_rows.append({
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "value": telemetry[feature],
            "unit": FEATURE_UNITS[feature],
            "state": max(membership, key=membership.get),
            "membership": round(float(max(membership.values())), 3),
            "contribution": round(float(contribution), 4),
        })
    return sorted(factor_rows, key=lambda row: row["contribution"], reverse=True)


app = FastAPI(title="Fuzzy Medical AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = FuzzyEngine()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
def predict(patient: PatientTelemetry) -> dict:
    telemetry = patient.model_dump()
    artifact = _load_artifact()
    fuzzy_result = engine.infer(telemetry)
    learned_score = float(artifact["model"].predict([telemetry])[0])
    combined_score = round(0.72 * learned_score + 0.28 * fuzzy_result["score"], 1)
    category = "Low" if combined_score < 33 else "Moderate" if combined_score < 66 else "High"
    return {
        "overall_risk": combined_score,
        "category": category,
        "model_risk": round(learned_score, 1),
        "fuzzy_risk": fuzzy_result["score"],
        "fuzzy_memberships": fuzzy_result["memberships"],
        "rule_activations": fuzzy_result["rule_activations"],
        "key_contributing_factors": _factors(telemetry, artifact, fuzzy_result),
        "model_metrics": artifact.get("metrics", {}),
    }


@app.post("/feedback")
def feedback(record: FeedbackRecord) -> dict:
    values = record.model_dump()
    created_at = datetime.now(timezone.utc).isoformat()
    connection = _connect()
    try:
        cursor = connection.execute(
            "INSERT INTO feedback (age, blood_pressure, cholesterol, blood_sugar, heart_rate, risk_score, verified, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*[values[name] for name in FEATURE_NAMES], values["risk_score"], int(values["verified"]), values["source"], created_at),
        )
        connection.commit()
        return {"status": "stored", "feedback_id": cursor.lastrowid, "verified": values["verified"], "source": values["source"], "created_at": created_at}
    finally:
        connection.close()


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
