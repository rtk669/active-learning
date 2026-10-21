"""Стратегии выбора: им доступны только ID и предсказания, но не метки."""

import numpy as np

from data import Array


def select_positions(
    strategy: str, pool_size: int, count: int, rng: np.random.Generator,
    probabilities: Array | None = None,
) -> Array:
    """Возвращает позиции выбранных примеров в текущем пуле без повторов.

    Для двух классов минимальная уверенность соответствует p ≈ 0.5.
    Сортируем |p - 0.5| по возрастанию: p=0.51 раньше p=0.9.
    При равных расстояниях случайная перестановка устраняет предпочтение
    первых строк. Истинные классы не участвуют ни в одной из операций.
    """
    if not 0 < count <= pool_size:
        raise ValueError("Размер запроса должен быть от 1 до размера пула")
    if strategy == "random":
        return rng.choice(pool_size, size=count, replace=False)
    if strategy != "uncertainty":
        raise ValueError(f"Неизвестная стратегия: {strategy}")
    if probabilities is None:
        raise ValueError("Для uncertainty нужны вероятности")
    probabilities = np.asarray(probabilities).reshape(-1)
    if len(probabilities) != pool_size or not np.all(np.isfinite(probabilities)):
        raise ValueError("Некорректные предсказания пула")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Вероятности должны лежать в [0, 1]")
    order = rng.permutation(pool_size)
    ranked = np.argsort(np.abs(probabilities[order] - 0.5), kind="stable")
    return order[ranked[:count]]


def remove_selected(pool_ids: Array, positions: Array) -> Array:
    """Удаляет выбранные ID: следующий цикл физически не сможет выбрать их снова."""
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Повторяющиеся позиции")
    if np.any(positions < 0) or np.any(positions >= len(pool_ids)):
        raise ValueError("Позиция вне пула")
    keep = np.ones(len(pool_ids), dtype=bool)
    keep[positions] = False
    return pool_ids[keep]
