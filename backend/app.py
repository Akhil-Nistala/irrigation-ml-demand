"""
Flask backend for the Irrigation Water Demand predictor.

Serves the static frontend (../frontend) and a small JSON API:
  GET  /api/meta      -> feature ranges/units, model comparison metrics, feature importances
  POST /api/predict   -> {features...} -> predicted irrigation_demand_mm + per-feature
                          contribution breakdown (a from-scratch tree-path decomposition
                          of the Random Forest prediction, in the spirit of treeinterpreter)

Run with:  python backend/app.py
Then open http://localhost:5000
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "irrigation_dataset.csv"
FIGURES_DIR = ROOT_DIR / "figures"
IMPORTANCE_CSV = ROOT_DIR / "report" / "feature_importance_summary.csv"

FEATURES = [
    "rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps",
    "solar_radiation_MJ_m2_day", "previous_irrigation_mm", "crop_coefficient", "growth_stage",
]

FEATURE_META = {
    "rainfall_mm": {"label": "Rainfall", "unit": "mm/day"},
    "temperature_C": {"label": "Mean temperature", "unit": "°C"},
    "humidity_percent": {"label": "Relative humidity", "unit": "%"},
    "wind_speed_mps": {"label": "Wind speed", "unit": "m/s"},
    "solar_radiation_MJ_m2_day": {"label": "Solar radiation", "unit": "MJ/m²/day"},
    "previous_irrigation_mm": {"label": "Previous day's irrigation", "unit": "mm"},
    "crop_coefficient": {"label": "Crop coefficient (Kc)", "unit": ""},
    "growth_stage": {"label": "Growth stage", "unit": "1=initial, 2=development, 3=mid, 4=late"},
}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# ---- Load once at startup ------------------------------------------------
model = joblib.load(MODELS_DIR / "random_forest.joblib")
with open(MODELS_DIR / "metrics.json") as f:
    metrics = json.load(f)
importance_df = pd.read_csv(IMPORTANCE_CSV, index_col=0)
dataset = pd.read_csv(DATA_PATH)

FEATURE_STATS = {
    col: {
        "min": float(dataset[col].min()),
        "max": float(dataset[col].max()),
        "mean": float(dataset[col].mean()),
    }
    for col in FEATURES
}


def tree_contributions(rf, x_row):
    """Decompose a RandomForest prediction into a bias term + per-feature
    contributions by walking the decision path of every tree and attributing
    the change in the node's mean prediction to the feature that split it.

    bias + sum(contributions) == rf.predict(x_row) (up to floating point error).
    """
    x = x_row.reshape(1, -1).astype(np.float32)
    n_features = x_row.shape[0]
    contributions = np.zeros(n_features)
    bias = 0.0

    for tree in rf.estimators_:
        t = tree.tree_
        path = tree.decision_path(x).indices
        values = t.value[path, 0, 0]
        bias += values[0]
        for i in range(len(path) - 1):
            feat_idx = t.feature[path[i]]
            contributions[feat_idx] += values[i + 1] - values[i]

    n_estimators = len(rf.estimators_)
    return bias / n_estimators, contributions / n_estimators


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "features": [
            {
                "name": name,
                "label": FEATURE_META[name]["label"],
                "unit": FEATURE_META[name]["unit"],
                **FEATURE_STATS[name],
            }
            for name in FEATURES
        ],
        "model_comparison": metrics,
        "active_model": "Random Forest",
        "feature_importance": {
            "features": list(importance_df.index),
            "rf_mdi_importance": importance_df["rf_mdi_importance"].round(4).tolist(),
            "rf_permutation_importance_mean": importance_df["rf_permutation_importance_mean"].round(4).tolist(),
            "correlation_with_target": importance_df["correlation_with_target"].round(4).tolist(),
        },
        "dataset_size": len(dataset),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        x_row = np.array([float(payload[f]) for f in FEATURES])
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric."}), 400

    x_df = pd.DataFrame([x_row], columns=FEATURES)
    prediction = float(model.predict(x_df)[0])
    prediction = max(prediction, 0.0)

    bias, contributions = tree_contributions(model, x_row)

    breakdown = sorted(
        (
            {
                "feature": FEATURES[i],
                "label": FEATURE_META[FEATURES[i]]["label"],
                "value": float(x_row[i]),
                "contribution": float(contributions[i]),
                "dataset_mean": FEATURE_STATS[FEATURES[i]]["mean"],
            }
            for i in range(len(FEATURES))
        ),
        key=lambda d: abs(d["contribution"]),
        reverse=True,
    )

    return jsonify({
        "prediction_mm": round(prediction, 3),
        "bias_mm": round(float(bias), 3),
        "breakdown": breakdown,
        "model": "Random Forest (300 trees, max_depth=10)",
    })


@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
