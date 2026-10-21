"""Настройки двух режимов эксперимента."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Размеры наборов и общие для стратегий параметры обучения."""

    mode: str = "quick"
    seed: int = 42
    initial_size: int = 500
    pool_size: int = 2500
    validation_size: int = 1000
    test_size: int = 2000
    cycles: int = 2
    query_size: int = 250
    epochs: int = 3
    batch_size: int = 64
    vocab_size: int = 10000
    max_length: int = 200
    embedding_dim: int = 32
    learning_rate: float = 0.001

    def validate(self) -> None:
        """Проверяет бюджет разметки и размеры официальных частей IMDB."""
        for name in (
            "initial_size", "pool_size", "validation_size", "test_size",
            "query_size", "epochs", "batch_size", "vocab_size", "max_length",
            "embedding_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть положительным")
        if self.cycles < 0 or not 0 <= self.seed < 2**32 - 1 - self.cycles:
            raise ValueError("Некорректное число циклов или seed")
        if not 0 < self.learning_rate < 1 or self.vocab_size < 4:
            raise ValueError("Проверьте learning_rate и vocab_size")
        if self.initial_size + self.pool_size + self.validation_size > 25000:
            raise ValueError("Начальная выборка + пул + validation превышают 25000")
        if self.test_size > 25000:
            raise ValueError("В официальной тестовой части только 25000 отзывов")
        if self.cycles * self.query_size > self.pool_size:
            raise ValueError("В пуле недостаточно примеров для указанного бюджета")


def preset(mode: str) -> Config:
    """Возвращает быстрый или полноценный режим."""
    if mode == "quick":
        return Config()
    if mode == "full":
        return Config(
            mode="full", initial_size=1000, pool_size=19000,
            validation_size=5000, test_size=25000, cycles=8,
            query_size=1000, epochs=10, max_length=250,
        )
    raise ValueError(f"Неизвестный режим: {mode}")
