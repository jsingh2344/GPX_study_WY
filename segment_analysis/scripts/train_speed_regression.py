#!/usr/bin/env python3
"""Predict 40 m segment speed from slope and elevation with linear regression."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "slope_speed_40m_segments_cleaned_gpx.csv"
PREDICTIONS = ROOT / "data" / "speed_regression_predictions.csv"
PLOT = ROOT / "products" / "speed_regression_actual_vs_predicted.png"
TEST_FRACTION = 0.20
RANDOM_SEED = 7


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "file": row["file"],
                    "slope": float(row["slope_angle_deg"]),
                    "elevation": float(row["midpoint_elevation_ft"]),
                    "speed": float(row["speed_mph"]),
                }
            )
    return rows


def split_by_file(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str]]:
    files = sorted({str(row["file"]) for row in rows})
    random.Random(RANDOM_SEED).shuffle(files)
    test_files = set(files[: max(1, round(len(files) * TEST_FRACTION))])
    train = [row for row in rows if row["file"] not in test_files]
    test = [row for row in rows if row["file"] in test_files]
    return train, test, test_files


def feature_stats(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array([[row["slope"], row["elevation"]] for row in rows], dtype=float)
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    return means, stds


def design_matrix(rows: list[dict[str, object]], means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    features = np.array([[row["slope"], row["elevation"]] for row in rows], dtype=float)
    scaled = (features - means) / stds
    return np.column_stack([np.ones(len(rows)), scaled])


def speeds(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([row["speed"] for row in rows], dtype=float)


def predict(rows: list[dict[str, object]], coefficients: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    predictions = []
    for row in rows:
        scaled_slope = (row["slope"] - means[0]) / stds[0]
        scaled_elevation = (row["elevation"] - means[1]) / stds[1]
        predictions.append(coefficients[0] + coefficients[1] * scaled_slope + coefficients[2] * scaled_elevation)
    return np.array(predictions, dtype=float)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    errors = predicted - actual
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    r2 = float(1 - np.sum(errors**2) / np.sum((actual - actual.mean()) ** 2))
    return rmse, mae, r2


def write_predictions(rows: list[dict[str, object]], predicted: np.ndarray) -> None:
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "slope_angle_deg", "midpoint_elevation_ft", "actual_speed_mph", "predicted_speed_mph"])
        for row, pred in zip(rows, predicted):
            writer.writerow([row["file"], row["slope"], row["elevation"], row["speed"], round(float(pred), 3)])


def plot_predictions(actual: np.ndarray, predicted: np.ndarray) -> None:
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.scatter(actual, predicted, s=10, alpha=0.35, edgecolors="none")
    limit = max(float(actual.max()), float(predicted.max()))
    ax.plot([0, limit], [0, limit], color="black", linewidth=1.5)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Actual speed (mph)")
    ax.set_ylabel("Predicted speed (mph)")
    ax.set_title("Linear Regression: Actual vs. Predicted Segment Speed")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT)
    plt.close(fig)


def main() -> int:
    rows = read_rows(DATA)
    train, test, test_files = split_by_file(rows)

    means, stds = feature_stats(train)
    coefficients = np.linalg.lstsq(design_matrix(train, means, stds), speeds(train), rcond=None)[0]
    predicted = predict(test, coefficients, means, stds)
    rmse, mae, r2 = metrics(speeds(test), predicted)
    original_slope_coef = coefficients[1] / stds[0]
    original_elevation_coef = coefficients[2] / stds[1]
    original_intercept = coefficients[0] - original_slope_coef * means[0] - original_elevation_coef * means[1]

    write_predictions(test, predicted)
    plot_predictions(speeds(test), predicted)

    print(f"Rows: {len(rows):,} total, {len(train):,} train, {len(test):,} test")
    print(f"Test files: {', '.join(sorted(test_files))}")
    print(f"Model: speed = {original_intercept:.4f} + {original_slope_coef:.4f}*slope + {original_elevation_coef:.6f}*elevation")
    print(f"Test RMSE: {rmse:.3f} mph")
    print(f"Test MAE: {mae:.3f} mph")
    print(f"Test R^2: {r2:.3f}")
    print(f"Wrote {PREDICTIONS}")
    print(f"Wrote {PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
