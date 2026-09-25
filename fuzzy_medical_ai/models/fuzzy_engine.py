"""Explainable five-state fuzzy inference with U-shaped clinical risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import skfuzzy as fuzz

FEATURE_NAMES = ["age", "blood_pressure", "cholesterol", "blood_sugar", "heart_rate"]
FEATURE_LABELS = {"age": "Age", "blood_pressure": "Blood pressure", "cholesterol": "Cholesterol", "blood_sugar": "Blood sugar", "heart_rate": "Heart rate"}
FEATURE_UNITS = {"age": "years", "blood_pressure": "mmHg", "cholesterol": "mg/dL", "blood_sugar": "mg/dL", "heart_rate": "bpm"}
FUZZY_STATES = ("Critically_Low", "Low_Normal", "Optimal", "High_Normal", "Critically_High")
HEALTHY_BASELINES = {"age": 30.0, "blood_pressure": 120.0, "cholesterol": 170.0, "blood_sugar": 90.0, "heart_rate": 72.0}
DEVIATION_SCALES = {"age": 70.0, "blood_pressure": 80.0, "cholesterol": 190.0, "blood_sugar": 210.0, "heart_rate": 78.0}


@dataclass(frozen=True)
class MembershipDefinition:
    minimum: float
    maximum: float
    critically_low: tuple[float, float, float, float]
    low_normal: tuple[float, float, float, float]
    optimal_center: float
    optimal_sigma: float
    high_normal: tuple[float, float, float, float]
    critically_high: tuple[float, float, float, float]


MEMBERSHIP_DEFINITIONS = {
    "age": MembershipDefinition(18, 100, (18, 18, 20, 23), (20, 23, 27, 30), 30, 5, (32, 38, 52, 65), (60, 70, 100, 100)),
    "blood_pressure": MembershipDefinition(80, 200, (80, 80, 84, 90), (86, 92, 105, 116), 120, 7, (124, 132, 148, 160), (156, 164, 200, 200)),
    "cholesterol": MembershipDefinition(100, 360, (100, 100, 108, 120), (112, 125, 150, 165), 170, 13, (178, 190, 220, 245), (235, 250, 360, 360)),
    "blood_sugar": MembershipDefinition(55, 300, (55, 55, 60, 70), (65, 72, 82, 88), 90, 8, (96, 110, 145, 180), (172, 185, 300, 300)),
    "heart_rate": MembershipDefinition(40, 150, (40, 40, 44, 52), (48, 55, 64, 70), 72, 6, (76, 84, 102, 115), (108, 118, 150, 150)),
}


class FuzzyEngine:
    """Convert telemetry into five memberships and transparent risk rules."""

    def _membership(self, feature: str, value: float) -> dict[str, float]:
        definition = MEMBERSHIP_DEFINITIONS[feature]
        clipped = float(np.clip(value, definition.minimum, definition.maximum))
        universe = np.linspace(definition.minimum, definition.maximum, 1201)
        curves = {
            "Critically_Low": fuzz.trapmf(universe, definition.critically_low),
            "Low_Normal": fuzz.trapmf(universe, definition.low_normal),
            "Optimal": fuzz.gaussmf(universe, definition.optimal_center, definition.optimal_sigma),
            "High_Normal": fuzz.trapmf(universe, definition.high_normal),
            "Critically_High": fuzz.trapmf(universe, definition.critically_high),
        }
        return {state: round(float(fuzz.interp_membership(universe, curve, clipped)), 4) for state, curve in curves.items()}

    def memberships(self, telemetry: dict[str, float]) -> dict[str, dict[str, float]]:
        return {feature: self._membership(feature, telemetry[feature]) for feature in FEATURE_NAMES}

    def infer(self, telemetry: dict[str, float]) -> dict[str, Any]:
        memberships = self.memberships(telemetry)
        critical_low = np.array([memberships[name]["Critically_Low"] for name in FEATURE_NAMES])
        critical_high = np.array([memberships[name]["Critically_High"] for name in FEATURE_NAMES])
        high_normal = np.array([memberships[name]["High_Normal"] for name in FEATURE_NAMES])
        low_normal = np.array([memberships[name]["Low_Normal"] for name in FEATURE_NAMES])
        critical_activation = float(max(np.max(critical_low), np.max(critical_high)))
        critical_alignment = float(np.mean(np.maximum(critical_low, critical_high)))
        normal_deviation = float(np.mean(np.maximum(low_normal, high_normal)))
        protective_alignment = float(np.mean([memberships[name]["Optimal"] for name in FEATURE_NAMES]))
        score = 100 * (0.72 * critical_activation + 0.18 * critical_alignment + 0.16 * normal_deviation - 0.12 * protective_alignment)
        if critical_activation >= 0.72:
            score = max(score, 82 + 18 * critical_alignment)
        score = float(np.clip(score, 0, 100))
        category = "Low" if score < 33 else "Moderate" if score < 66 else "High"
        return {"score": round(score, 1), "category": category, "memberships": memberships, "rule_activations": {"critical_low": round(float(np.max(critical_low)), 4), "critical_high": round(float(np.max(critical_high)), 4), "critical_metric_alignment": round(critical_alignment, 4), "normal_range_deviation": round(normal_deviation, 4), "optimal_range_alignment": round(protective_alignment, 4)}}

    def u_shaped_features(self, telemetry: dict[str, float]) -> list[float]:
        return [float(np.clip(abs(telemetry[name] - HEALTHY_BASELINES[name]) / DEVIATION_SCALES[name], 0, 1)) for name in FEATURE_NAMES]

    def feature_vector(self, telemetry: dict[str, float]) -> list[float]:
        memberships = self.memberships(telemetry)
        return [memberships[feature][state] for feature in FEATURE_NAMES for state in FUZZY_STATES]
