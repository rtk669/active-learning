"""Цикл обучения с одинаковым бюджетом и независимой итоговой оценкой."""

import logging
from pathlib import Path

import numpy as np
from tensorflow import keras

from active_learning import remove_selected, select_positions
from config import Config
from data import Array, Dataset
from evaluation import calculate_metrics, save_csv
from model import build_model

LOGGER = logging.getLogger(__name__)


def train_stage(config: Config, x: Array, y: Array, data: Dataset,
                stage: int) -> tuple[keras.Model, dict]:
    """Обучает с нуля фиксированное число эпох; validation не обновляет веса."""
    model = build_model(config, seed=config.seed + stage)
    history = model.fit(
        x, y, validation_data=(data.x_validation, data.y_validation),
        epochs=config.epochs, batch_size=config.batch_size, shuffle=True, verbose=0,
    )
    return model, history.history


def run_strategy(strategy: str, config: Config, data: Dataset,
                 output: Path) -> tuple[list[dict], keras.Model]:
    """Раскрывает новые метки только после выбора; сохраняет аудит запросов."""
    x_labeled, y_labeled = data.x_initial.copy(), data.y_initial.copy()
    # Здесь индексы строк исходного пула, а не исходные ID отзывов.
    remaining = np.arange(len(data.pool_ids))
    rng = np.random.default_rng(config.seed)
    rows, selections, histories = [], [], []
    for stage in range(config.cycles + 1):
        LOGGER.info("%s: этап %d/%d, размечено %d", strategy, stage, config.cycles, len(y_labeled))
        model, history = train_stage(config, x_labeled, y_labeled, data, stage)
        probabilities = model.predict(data.x_validation, batch_size=config.batch_size, verbose=0)
        metrics = calculate_metrics(data.y_validation, probabilities)
        rows.append(dict(strategy=strategy, stage=stage, labeled_count=len(y_labeled),
                         split="validation", **metrics))
        LOGGER.info("validation: accuracy=%.4f, F1=%.4f", metrics["accuracy"], metrics["f1"])
        for epoch in range(config.epochs):
            histories.append(dict(strategy=strategy, stage=stage, epoch=epoch + 1,
                                  **{key: values[epoch] for key, values in history.items()}))
        save_csv(output / f"{strategy}_validation.csv", rows)
        save_csv(output / f"{strategy}_history.csv", histories)
        if stage == config.cycles:
            break

        # Random не использует модель. Uncertainty видит только x и p(y=1|x).
        pool_probabilities = None
        if strategy == "uncertainty":
            pool_probabilities = model.predict(
                data.x_pool[remaining], batch_size=config.batch_size, verbose=0,
            ).reshape(-1)
        positions = select_positions(strategy, len(remaining), config.query_size,
                                     rng, pool_probabilities)
        selected_rows = remaining[positions]
        selected_ids = data.pool_ids[selected_rows]
        # Только теперь, после окончательного выбора, «человек» сообщает классы.
        new_labels = data.oracles[strategy].annotate(selected_ids)
        x_labeled = np.concatenate([x_labeled, data.x_pool[selected_rows]])
        y_labeled = np.concatenate([y_labeled, new_labels])
        for index, original_id in enumerate(selected_ids):
            selections.append(dict(
                strategy=strategy, acquired_after_stage=stage, sample_id=int(original_id),
                probability_before_annotation=(float(pool_probabilities[positions[index]])
                                               if pool_probabilities is not None else ""),
                revealed_label=int(new_labels[index]),
            ))
        remaining = remove_selected(remaining, positions)
        save_csv(output / f"{strategy}_queries.csv", selections)
    return rows, model


def run_experiment(config: Config, data: Dataset, output: Path) -> tuple[list[dict], list[dict]]:
    """Завершает обе стратегии прежде, чем обращаться к тестовым данным."""
    validation_rows, final_models = [], {}
    for strategy in ("random", "uncertainty"):
        rows, final_models[strategy] = run_strategy(strategy, config, data, output)
        validation_rows.extend(rows)
    test_rows = []
    for strategy, model in final_models.items():
        probabilities = model.predict(data.x_test, batch_size=config.batch_size, verbose=0)
        test_rows.append(dict(
            strategy=strategy, stage=config.cycles,
            labeled_count=config.initial_size + config.cycles * config.query_size,
            split="test", **calculate_metrics(data.y_test, probabilities),
        ))
    save_csv(output / "validation_metrics.csv", validation_rows)
    save_csv(output / "test_metrics.csv", test_rows)
    return validation_rows, test_rows
