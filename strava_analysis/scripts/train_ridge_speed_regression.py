#!/usr/bin/env python3
"""Leave-one-track-out ridge regression for Strava segment speed."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "strava_slope_speed_40m_segments.csv"
RESULTS = ROOT / "data" / "strava_ridge_cv_results.csv"
PREDICTIONS = ROOT / "data" / "strava_ridge_cv_predictions.csv"
COEFFICIENTS = ROOT / "data" / "strava_ridge_coefficients.csv"
PLOT = ROOT / "products" / "strava_ridge_actual_vs_predicted.png"
MIN_SPEED_MPH = 0.2
ALPHAS = [0, 0.01, 0.1, 1, 10, 100, 1000]
SLOPE_BINS = [(-40, -30), (-30, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30), (30, 40)]


def read_rows() -> list[dict[str, object]]:
    rows = []
    with DATA.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            slope = float(row["slope_angle_deg"])
            elevation = float(row["midpoint_elevation_ft"])
            speed = float(row["speed_mph"])
            if speed < MIN_SPEED_MPH:
                continue
            rows.append({"file": row["file"], "slope": slope, "elevation": elevation, "speed": speed})
    return rows


def feature_names() -> list[str]:
    names = ["elevation", "elevation_x_uphill_slope"]
    names += [f"slope_bin_{low}_to_{high}" for low, high in SLOPE_BINS]
    return names


def features(row: dict[str, object]) -> list[float]:
    slope = float(row["slope"])
    elevation = float(row["elevation"])
    values = [elevation, elevation * max(slope, 0.0)]
    values += [1.0 if low <= slope < high else 0.0 for low, high in SLOPE_BINS]
    return values


def matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([features(row) for row in rows], dtype=float)


def speeds(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([row["speed"] for row in rows], dtype=float)


def log_speeds(rows: list[dict[str, object]]) -> np.ndarray:
    return np.log(speeds(rows))


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    return (train_x - means) / stds, (test_x - means) / stds, means, stds


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    penalty = np.sqrt(alpha) * np.eye(x_with_intercept.shape[1])
    penalty[0, 0] = 0.0
    augmented_x = np.vstack([x_with_intercept, penalty])
    augmented_y = np.concatenate([y, np.zeros(x_with_intercept.shape[1])])
    return np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.array([coefficients[0] + sum(coefficients[i + 1] * row[i] for i in range(len(row))) for row in x])


def score(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    errors = predicted - actual
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    r2 = float(1 - np.sum(errors**2) / np.sum((actual - actual.mean()) ** 2))
    return rmse, mae, r2


def cross_validate(rows: list[dict[str, object]], alpha: float) -> tuple[float, float, float, list[dict[str, object]]]:
    predictions = []
    for test_file in sorted({row["file"] for row in rows}):
        train = [row for row in rows if row["file"] != test_file]
        test = [row for row in rows if row["file"] == test_file]
        train_x, test_x, _, _ = standardize(matrix(train), matrix(test))
        coefficients = fit_ridge(train_x, log_speeds(train), alpha)
        predicted = np.exp(predict(test_x, coefficients))
        for row, pred in zip(test, predicted):
            predictions.append({**row, "predicted": float(pred), "alpha": alpha})
    actual = np.array([row["speed"] for row in predictions], dtype=float)
    predicted = np.array([row["predicted"] for row in predictions], dtype=float)
    return (*score(actual, predicted), predictions)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_coefficients(rows: list[dict[str, object]], alpha: float) -> None:
    x = matrix(rows)
    y = log_speeds(rows)
    scaled_x, _, means, stds = standardize(x, x)
    coefficients = fit_ridge(scaled_x, y, alpha)
    original = coefficients[1:] / stds
    intercept = coefficients[0] - sum(float(coef * mean) for coef, mean in zip(original, means))
    out_rows = [{"feature": "intercept", "coefficient": round(float(intercept), 8), "scale": "log_speed"}]
    out_rows += [
        {"feature": name, "coefficient": round(float(coef), 8), "scale": "log_speed"}
        for name, coef in zip(feature_names(), original)
    ]
    write_csv(COEFFICIENTS, out_rows, ["feature", "coefficient", "scale"])


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
    ax.set_title("Strava Log-Speed Ridge: Leave-One-Track-Out")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT)
    plt.close(fig)


def main() -> int:
    rows = read_rows()
    result_rows = []
    prediction_sets = {}
    for alpha in ALPHAS:
        rmse, mae, r2, predictions = cross_validate(rows, alpha)
        result_rows.append({"alpha": alpha, "rmse_mph": round(rmse, 4), "mae_mph": round(mae, 4), "r2": round(r2, 4)})
        prediction_sets[alpha] = predictions

    best = min(result_rows, key=lambda row: row["rmse_mph"])
    best_alpha = float(best["alpha"])
    best_predictions = prediction_sets[best_alpha]

    write_csv(RESULTS, result_rows, ["alpha", "rmse_mph", "mae_mph", "r2"])
    write_csv(
        PREDICTIONS,
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
    write_coefficients(rows, best_alpha)
    plot_predictions(best_predictions)

    print(f"Rows: {len(rows):,}; tracks: {len(set(row['file'] for row in rows))}; minimum speed: {MIN_SPEED_MPH:g} mph")
    print(f"Best alpha: {best_alpha:g}")
    print(f"RMSE: {best['rmse_mph']:.3f} mph; MAE: {best['mae_mph']:.3f} mph; R^2: {best['r2']:.3f}")
    print(f"Wrote {RESULTS}")
    print(f"Wrote {PREDICTIONS}")
    print(f"Wrote {COEFFICIENTS}")
    print(f"Wrote {PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
