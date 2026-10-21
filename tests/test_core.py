"""Проверки экспериментального протокола без загрузки IMDB и TensorFlow."""

from dataclasses import replace
import unittest

import numpy as np

from active_learning import remove_selected, select_positions
from config import Config, preset
from data import LabelOracle, split_train_indices
from evaluation import calculate_metrics


class SplitTests(unittest.TestCase):
    """Проверяет непересечение и воспроизводимость разбиения."""

    def test_disjoint_reproducible_splits(self) -> None:
        """Каждый ID встречается только в одном наборе."""
        config = Config(initial_size=10, pool_size=50, validation_size=20)
        first = split_train_indices(100, config)
        second = split_train_indices(100, config)
        self.assertEqual([len(part) for part in first], [10, 50, 20])
        joined = np.concatenate(first)
        self.assertEqual(len(np.unique(joined)), len(joined))
        self.assertTrue(np.all((joined >= 0) & (joined < 100)))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_budget_validation(self) -> None:
        """Невозможный бюджет отвергается до обучения."""
        for mode in ("quick", "full"):
            preset(mode).validate()
        with self.assertRaises(ValueError):
            replace(Config(), cycles=11).validate()
        with self.assertRaises(ValueError):
            split_train_indices(10, Config())


class SamplingTests(unittest.TestCase):
    """Проверяет отбор по неопределённости и расходование пула."""

    def test_uncertainty_nearest_half(self) -> None:
        """Выбираются p=0.5, 0.49, 0.52, а не уверенные предсказания."""
        chosen = select_positions("uncertainty", 5, 3, np.random.default_rng(1),
                                  np.array([0.01, 0.49, 0.99, 0.5, 0.52]))
        np.testing.assert_array_equal(chosen, [3, 1, 4])

    def test_both_strategies_exhaust_without_repeats(self) -> None:
        """После удаления ни одна стратегия не выбирает пример второй раз."""
        for strategy in ("random", "uncertainty"):
            rng = np.random.default_rng(5)
            remaining = np.arange(12)
            oracle = LabelOracle(remaining, remaining % 2)
            selected = []
            for _ in range(4):
                positions = select_positions(strategy, len(remaining), 3, rng,
                                             np.full(len(remaining), 0.5))
                ids = remaining[positions]
                np.testing.assert_array_equal(oracle.annotate(ids), ids % 2)
                selected.extend(ids.tolist())
                remaining = remove_selected(remaining, positions)
            self.assertEqual(sorted(selected), list(range(12)))
            self.assertEqual(len(remaining), 0)

    def test_oracle_rejects_duplicate_unknown_and_repeated_ids(self) -> None:
        """Неверный запрос атомарен: он не раскрывает часть меток."""
        oracle = LabelOracle(np.array([1, 2]), np.array([0, 1]))
        for invalid in ([1, 1], [1, 99]):
            with self.assertRaises(ValueError):
                oracle.annotate(np.array(invalid))
        np.testing.assert_array_equal(oracle.annotate(np.array([1])), [0])
        with self.assertRaises(ValueError):
            oracle.annotate(np.array([1]))
        np.testing.assert_array_equal(oracle.annotate(np.array([2])), [1])

    def test_invalid_predictions_and_budget(self) -> None:
        """Ошибки вероятностей не превращаются в произвольную выборку."""
        for probabilities in ([np.nan, 0.5], [1.1, 0.5], [0.5]):
            with self.assertRaises(ValueError):
                select_positions("uncertainty", 2, 1, np.random.default_rng(0),
                                 np.array(probabilities))
        with self.assertRaises(ValueError):
            select_positions("random", 2, 3, np.random.default_rng(0))

    def test_seeded_ties(self) -> None:
        """Одинаковые вероятности разрешаются воспроизводимо."""
        first = select_positions("uncertainty", 20, 5, np.random.default_rng(4), np.full(20, 0.5))
        second = select_positions("uncertainty", 20, 5, np.random.default_rng(4), np.full(20, 0.5))
        np.testing.assert_array_equal(first, second)


class MetricTests(unittest.TestCase):
    """Проверяет направление ошибок FP и FN."""

    def test_confusion_counts(self) -> None:
        """Положительный прогноз на отрицательном отзыве считается FP."""
        result = calculate_metrics(np.array([0, 0, 1, 1, 1]), np.array([0.1, 0.8, 0.2, 0.3, 0.9]))
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 2)
        self.assertAlmostEqual(result["accuracy"], 0.4)
        self.assertAlmostEqual(result["f1"], 0.4)


if __name__ == "__main__":
    unittest.main()
