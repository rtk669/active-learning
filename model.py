"""Небольшой классификатор отзывов на TensorFlow/Keras."""

from tensorflow import keras

from config import Config


def build_model(config: Config, seed: int) -> keras.Model:
    """Создаёт новую модель и оптимизатор с воспроизводимой инициализацией."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(config.max_length,), dtype="int32"),
        keras.layers.Embedding(config.vocab_size, config.embedding_dim, mask_zero=True),
        # Маска исключает padding из среднего: длина отзыва не разбавляет вектор.
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy", metrics=["accuracy"],
    )
    return model
