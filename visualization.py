"""Сохранение графиков без графического интерфейса."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_plots(validation: list[dict], test: list[dict], output: Path) -> None:
    """Рисует кривые validation и отдельное сравнение на итоговом test."""
    for metric, title in (("accuracy", "Accuracy"), ("f1", "F1-score")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in ("random", "uncertainty"):
            rows = [row for row in validation if row["strategy"] == strategy]
            ax.plot([r["labeled_count"] for r in rows], [r[metric] for r in rows],
                    marker="o", label=strategy)
        ax.set(xlabel="Labeled training reviews", ylabel=title, ylim=(0, 1),
               title=f"Validation {title} — equal annotation budget")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"validation_{metric}.png", dpi=160)
        plt.close(fig)
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(metrics))
    for index, row in enumerate(test):
        bars = ax.bar(positions + (index - 0.5) * 0.36,
                      [row[m] for m in metrics], width=0.36, label=row["strategy"])
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set(xticks=positions, xticklabels=metrics, ylim=(0, 1.12),
           ylabel="Score", title=f"Final independent test — {test[0]['labeled_count']} labeled reviews")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "final_test_comparison.png", dpi=160)
    plt.close(fig)
