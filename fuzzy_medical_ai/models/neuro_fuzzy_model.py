"""Nonlinear neuro-fuzzy regression over raw, fuzzy, and U-shaped features."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

from models.fuzzy_engine import FEATURE_NAMES, FUZZY_STATES, FuzzyEngine


class NeuroFuzzyRegressor:
    """Blend a nonlinear tree ensemble with spline-expanded U-shaped features."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.engine = FuzzyEngine()
        self.model = StackingRegressor(
            estimators=[
                ("gradient_boosting", GradientBoostingRegressor(n_estimators=240, learning_rate=0.035, max_depth=3, min_samples_leaf=5, loss="huber", random_state=random_state)),
                ("extra_trees", ExtraTreesRegressor(n_estimators=180, min_samples_leaf=4, max_features=0.85, random_state=random_state, n_jobs=-1)),
            ],
            final_estimator=Ridge(alpha=1.0),
            passthrough=True,
            n_jobs=-1,
        )
        self.feature_names = FEATURE_NAMES + [f"{feature}_{state.lower()}" for feature in FEATURE_NAMES for state in FUZZY_STATES] + [f"{feature}_u_distance" for feature in FEATURE_NAMES] + ["u_distance_mean", "critical_signal_count"]

    def _matrix(self, records: Iterable[dict[str, float]]) -> np.ndarray:
        rows = []
        for record in records:
            fuzzy_values = self.engine.feature_vector(record)
            u_distances = self.engine.u_shaped_features(record)
            critical_count = sum(value >= 0.65 for value in u_distances) / len(u_distances)
            rows.append([record[name] for name in FEATURE_NAMES] + fuzzy_values + u_distances + [float(np.mean(u_distances)), critical_count])
        return np.asarray(rows, dtype=float)

    def fit(self, records: Iterable[dict[str, float]], targets: Iterable[float]) -> "NeuroFuzzyRegressor":
        self.model.fit(self._matrix(records), np.asarray(list(targets), dtype=float))
        return self

    def predict(self, records: Iterable[dict[str, float]]) -> np.ndarray:
        records = list(records)
        learned_scores = self.model.predict(self._matrix(records))
        fuzzy_scores = np.asarray([self.engine.infer(record)["score"] for record in records])
        critical_guards = np.asarray([
            max(
                self.engine.infer(record)["memberships"][feature][state]
                for feature in FEATURE_NAMES
                for state in ("Critically_Low", "Critically_High")
            ) >= 0.72
            for record in records
        ])
        learned_scores = np.where(critical_guards, np.maximum(learned_scores, fuzzy_scores * 0.9), learned_scores)
        return np.clip(learned_scores, 0, 100)

    def feature_importances(self) -> dict[str, float]:
        importances = np.zeros(len(self.feature_names))
        for estimator in self.model.estimators_:
            if hasattr(estimator, "feature_importances_"):
                importances += estimator.feature_importances_
        if importances.sum() == 0:
            importances[0] = 1
        importances /= importances.sum()
        return {name: round(float(value), 4) for name, value in zip(self.feature_names, importances)}
