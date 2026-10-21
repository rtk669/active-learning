"""Подготовка IMDB и изоляция скрытых меток пула."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from config import Config

Array = NDArray[np.generic]


class LabelOracle:
    """Имитирует разметчика: выдаёт метки только запрошенных ID, один раз.

    Это граница интерфейса, а не защита от злонамеренного доступа в Python.
    Функции выбора вообще не получают объект разметчика или истинные метки.
    Для каждой стратегии создаётся независимый разметчик.
    """

    def __init__(self, ids: Array, labels: Array) -> None:
        if len(ids) != len(labels) or len(np.unique(ids)) != len(ids):
            raise ValueError("ID разметчика должны быть уникальны и соответствовать меткам")
        self.__labels = dict(zip(map(int, ids), map(int, labels)))

    def annotate(self, ids: Array) -> Array:
        """Раскрывает выбранные метки и запрещает повторную разметку."""
        keys = list(map(int, ids))
        if len(set(keys)) != len(keys) or any(k not in self.__labels for k in keys):
            raise ValueError("Повторный или неизвестный ID при запросе разметки")
        return np.array([self.__labels.pop(k) for k in keys], dtype=np.int32)


@dataclass
class Dataset:
    """Общие данные; истинные метки пула находятся только у разметчиков."""

    x_initial: Array
    y_initial: Array
    initial_ids: Array
    x_pool: Array
    pool_ids: Array
    x_validation: Array
    y_validation: Array
    validation_ids: Array
    x_test: Array
    y_test: Array
    test_ids: Array
    oracles: dict[str, LabelOracle]


def split_train_indices(size: int, config: Config) -> tuple[Array, Array, Array]:
    """Разделяет индексы без просмотра меток, включая начальную выборку.

    Случайный отбор не гарантирует точного баланса классов, зато не требует
    заранее знать скрытые классы. Неиспользованные отзывы просто отбрасываются.
    """
    needed = config.initial_size + config.pool_size + config.validation_size
    if needed > size:
        raise ValueError("Недостаточно обучающих отзывов")
    remainder, validation = train_test_split(
        np.arange(size), test_size=config.validation_size, random_state=config.seed,
    )
    shuffled = np.random.default_rng(config.seed).permutation(remainder)
    boundary = config.initial_size
    return shuffled[:boundary], shuffled[boundary:boundary + config.pool_size], validation


def load_data(config: Config) -> Dataset:
    """Загружает готовые токены Keras IMDB; сохраняет официальный test отдельно."""
    from tensorflow import keras

    # KERAS_HOME задаётся в main до импорта Keras: кэш остаётся внутри проекта.
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
            num_words=config.vocab_size, seed=config.seed,
        )
    except Exception as error:
        # Keras оборачивает сетевые ошибки в обычный Exception.
        raise RuntimeError(
            "Не удалось загрузить IMDB. Проверьте сеть, сертификаты и кэш "
            f".cache/keras/datasets. Причина: {error}"
        ) from error
    initial, pool, validation = split_train_indices(len(x_train), config)
    test = np.random.default_rng(config.seed + 1).permutation(len(x_test))[:config.test_size]

    def pad(sequences: Array) -> Array:
        """Дополняет нулями справа и оставляет начало каждого отзыва."""
        return keras.utils.pad_sequences(
            sequences, maxlen=config.max_length, padding="post", truncating="post",
        )

    return Dataset(
        pad(x_train[initial]), y_train[initial], initial,
        pad(x_train[pool]), pool,
        pad(x_train[validation]), y_train[validation], validation,
        pad(x_test[test]), y_test[test], test + len(x_train),
        {name: LabelOracle(pool, y_train[pool]) for name in ("random", "uncertainty")},
    )
