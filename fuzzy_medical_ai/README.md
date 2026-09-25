# Fuzzy Medical AI

Fuzzy Medical AI is a decision-support prototype for transparent health risk assessment. It combines explicit fuzzy membership reasoning with a Gradient Boosting regressor over raw telemetry and fuzzy features. The output is an analytical estimate for research and workflow prototyping, not a medical diagnosis.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\generate_mock_data.py
python train.py
python -m uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000` for the clinical workbench. The API is also available at `/predict`, `/feedback`, and `/health`.

## Workflow

1. `scripts/generate_mock_data.py` creates a reproducible CSV with five continuous health signals and a synthetic 0 to 100 target.
2. `train.py` adds fuzzy memberships to the raw measurements, trains the regressor, prints RMSE and R2, and atomically writes `models/saved_model.pkl`.
3. `api/app.py` validates incoming telemetry, combines learned and fuzzy scores, and returns memberships, rule activations, and ranked factors.
4. Verified feedback is written to `data/user_feedback.db`. Run `python api\retrain_pipeline.py` to evaluate a merged candidate and replace the live artifact only when its validation is at least as good as the current model.

## Model notes

Membership definitions are intentionally explicit in `models/fuzzy_engine.py`. Each input has Low, Moderate, and High trapezoidal sets. The learned model receives the five raw values plus fifteen membership values, preserving a direct path from the model output back to an interpretable signal.

## Feedback example

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/feedback -ContentType 'application/json' -Body (@{
  age=54; blood_pressure=128; cholesterol=214; blood_sugar=112; heart_rate=78;
  risk_score=41; verified=$true; source='doctor_review'
} | ConvertTo-Json)
```

The included dataset is synthetic. Do not send identifiable patient information to this development service.
