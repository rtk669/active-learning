"""Метрики бинарной классификации и сохранение таблиц."""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data import Array


def calculate_metrics(labels: Array, probabilities: Array) -> dict[str, float | int]:
    """Считает метрики; положительный класс 1 — положительный отзыв."""
    predicted = (np.asarray(probabilities).reshape(-1) >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision), recall=float(recall), f1=float(f1),
                false_positive=int(fp), false_negative=int(fn),
                true_positive=int(tp), true_negative=int(tn))


def save_csv(path: Path, rows: list[dict]) -> None:
    """Сохраняет таблицу с заголовком в UTF-8."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
