#!/usr/bin/env python3
"""Leave-one-track-out gradient boosting for cleaned GPX segment speed."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "slope_speed_40m_segments_cleaned_gpx.csv"
CV_RESULTS = ROOT / "data" / "gradient_boosting_cv_results.csv"
CV_PREDICTIONS = ROOT / "data" / "gradient_boosting_cv_predictions.csv"
IMPORTANCE = ROOT / "data" / "gradient_boosting_feature_importance.csv"
PLOT = ROOT / "products" / "gradient_boosting_actual_vs_predicted.png"

MIN_SPEED_MPH = 0.2
CONFIGS = [
    {"n_estimators": 40, "learning_rate": 0.08},
]
MAX_THRESHOLDS_PER_FEATURE = 16


FEATURE_NAMES = [
    "slope",
    "elevation",
    "abs_slope",
    "uphill_slope",
    "downhill_slope",
    "slope_squared",
    "elevation_x_uphill_slope",
]


@dataclass
class Stump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float
    improvement: float


def read_rows() -> list[dict[str, object]]:
    rows = []
    with DATA.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            speed = float(row["speed_mph"])
            if speed < MIN_SPEED_MPH:
                continue
            slope = float(row["slope_angle_deg"])
            elevation = float(row["midpoint_elevation_ft"])
            rows.append({"file": row["file"], "slope": slope, "elevation": elevation, "speed": speed})
    return rows


def features(row: dict[str, object]) -> list[float]:
    slope = float(row["slope"])
    elevation = float(row["elevation"])
    return [
        slope,
        elevation,
        abs(slope),
        max(slope, 0.0),
        max(-slope, 0.0),
        slope**2,
        elevation * max(slope, 0.0),
    ]


def matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([features(row) for row in rows], dtype=float)


def log_speeds(rows: list[dict[str, object]]) -> np.ndarray:
    return np.log(np.array([row["speed"] for row in rows], dtype=float))


def speeds(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([row["speed"] for row in rows], dtype=float)


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= MAX_THRESHOLDS_PER_FEATURE:
        return (unique[:-1] + unique[1:]) / 2
    quantiles = np.linspace(0.02, 0.98, MAX_THRESHOLDS_PER_FEATURE)
    return np.unique(np.quantile(values, quantiles))


def fit_best_stump(x: np.ndarray, residuals: np.ndarray) -> Stump:
    base_error = float(np.sum(residuals**2))
    best = Stump(0, 0.0, 0.0, 0.0, -math.inf)

    for feature_index in range(x.shape[1]):
        values = x[:, feature_index]
        for threshold in candidate_thresholds(values):
            left = values <= threshold
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            left_value = float(residuals[left].mean())
            right_value = float(residuals[right].mean())
            fitted = np.where(left, left_value, right_value)
            error = float(np.sum((residuals - fitted) ** 2))
            improvement = base_error - error
            if improvement > best.improvement:
                best = Stump(feature_index, float(threshold), left_value, right_value, improvement)

    return best


def stump_predict(x: np.ndarray, stump: Stump) -> np.ndarray:
    return np.where(x[:, stump.feature_index] <= stump.threshold, stump.left_value, stump.right_value)


def fit_boosting(x: np.ndarray, y: np.ndarray, n_estimators: int, learning_rate: float) -> tuple[float, list[Stump]]:
    intercept = float(y.mean())
    current = np.full(len(y), intercept)
    stumps = []
    for _ in range(n_estimators):
        residuals = y - current
        stump = fit_best_stump(x, residuals)
        current += learning_rate * stump_predict(x, stump)
        stumps.append(stump)
    return intercept, stumps


def predict_boosting(x: np.ndarray, intercept: float, stumps: list[Stump], learning_rate: float) -> np.ndarray:
    pred = np.full(len(x), intercept)
    for stump in stumps:
        pred += learning_rate * stump_predict(x, stump)
    return np.exp(pred)


def score(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    errors = predicted - actual
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    r2 = float(1 - np.sum(errors**2) / np.sum((actual - actual.mean()) ** 2))
    return rmse, mae, r2


def cross_validate(rows: list[dict[str, object]], config: dict[str, float]) -> tuple[float, float, float, list[dict[str, object]]]:
    predictions = []
    for test_file in sorted({row["file"] for row in rows}):
        train = [row for row in rows if row["file"] != test_file]
        test = [row for row in rows if row["file"] == test_file]
        intercept, stumps = fit_boosting(
            matrix(train),
            log_speeds(train),
            int(config["n_estimators"]),
            float(config["learning_rate"]),
        )
        predicted = predict_boosting(matrix(test), intercept, stumps, float(config["learning_rate"]))
        for row, pred in zip(test, predicted):
            predictions.append({**row, "predicted": float(pred)})

    actual = np.array([row["speed"] for row in predictions], dtype=float)
    predicted = np.array([row["predicted"] for row in predictions], dtype=float)
    return (*score(actual, predicted), predictions)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_importance(rows: list[dict[str, object]], config: dict[str, float]) -> None:
    intercept, stumps = fit_boosting(
        matrix(rows),
        log_speeds(rows),
        int(config["n_estimators"]),
        float(config["learning_rate"]),
    )
    totals = {name: 0.0 for name in FEATURE_NAMES}
    for stump in stumps:
        totals[FEATURE_NAMES[stump.feature_index]] += max(0.0, stump.improvement)
    total = sum(totals.values()) or 1.0
    out = [
        {"feature": name, "relative_importance": round(value / total, 6)}
        for name, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]
    write_csv(IMPORTANCE, out, ["feature", "relative_importance"])
    _ = intercept


def plot_predictions(rows: list[dict[str, object]]) -> None:
    actual = np.array([row["speed"] for row in rows], dtype=float)
    predicted = np.array([row["predicted"] for row in rows], dtype=float)
    limit = max(float(actual.max()), float(predicted.max()))
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.scatter(actual, predicted, s=8, alpha=0.28, edgecolors="none")
    ax.plot([0, limit], [0, limit], color="black", linewidth=1.4)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Actual speed (mph)")
    ax.set_ylabel("Predicted speed (mph)")
    ax.set_title("Gradient Boosting: Leave-One-Track-Out Predictions")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT)
    plt.close(fig)


def main() -> int:
    rows = read_rows()
    cv_rows = []
    prediction_sets = {}
    for config in CONFIGS:
        rmse, mae, r2, predictions = cross_validate(rows, config)
        key = f"{config['n_estimators']} trees, lr={config['learning_rate']}"
        cv_rows.append(
            {
                "model": key,
                "n_estimators": config["n_estimators"],
                "learning_rate": config["learning_rate"],
                "rmse_mph": round(rmse, 4),
                "mae_mph": round(mae, 4),
                "r2": round(r2, 4),
            }
        )
        prediction_sets[key] = predictions

    best = min(cv_rows, key=lambda row: row["rmse_mph"])
    best_key = str(best["model"])
    best_predictions = prediction_sets[best_key]

    write_csv(CV_RESULTS, cv_rows, ["model", "n_estimators", "learning_rate", "rmse_mph", "mae_mph", "r2"])
    write_csv(
        CV_PREDICTIONS,
        [
            {
                "file": row["file"],
                "slope_angle_deg": row["slope"],
                "midpoint_elevation_ft": row["elevation"],
                "actual_speed_mph": row["speed"],
                "predicted_speed_mph": round(float(row["predicted"]), 3),
            }
            for row in best_predictions
        ],
        ["file", "slope_angle_deg", "midpoint_elevation_ft", "actual_speed_mph", "predicted_speed_mph"],
    )
    best_config = next(config for config in CONFIGS if f"{config['n_estimators']} trees, lr={config['learning_rate']}" == best_key)
    write_importance(rows, best_config)
    plot_predictions(best_predictions)

    print(f"Rows: {len(rows):,}; tracks: {len(set(row['file'] for row in rows))}; minimum speed: {MIN_SPEED_MPH:g} mph")
    print(f"Best model: {best_key}")
    print(f"Leave-one-track-out RMSE: {best['rmse_mph']:.3f} mph")
    print(f"Leave-one-track-out MAE: {best['mae_mph']:.3f} mph")
    print(f"Leave-one-track-out R^2: {best['r2']:.3f}")
    print(f"Wrote {CV_RESULTS}")
    print(f"Wrote {CV_PREDICTIONS}")
    print(f"Wrote {IMPORTANCE}")
    print(f"Wrote {PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
