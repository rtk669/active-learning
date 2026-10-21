"""Командная строка учебного эксперимента Active Learning."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform

from config import Config, preset


def parse_args() -> argparse.Namespace:
    """Позволяет переопределить настройки выбранного режима."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, help="Новая папка запуска (не должна существовать)")
    for name, value in asdict(Config()).items():
        if name != "mode":
            parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=None)
    return parser.parse_args()


def main() -> int:
    """Настраивает CPU, запускает эксперимент и записывает его параметры."""
    args = parse_args()
    config = replace(preset(args.mode), **{
        key: value for key, value in vars(args).items()
        if key not in ("mode", "output") and value is not None
    })
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    root = Path(__file__).resolve().parent
    cache = root / ".cache"
    # Устанавливаем до импорта TensorFlow/Matplotlib, включая их транзитивные импорты.
    os.environ["KERAS_HOME"] = str(cache / "keras")
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        config.validate()
        output = args.output or root / "results" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output.mkdir(parents=True, exist_ok=False)
        cache.mkdir(parents=True, exist_ok=True)
        import tensorflow as tf
        import sklearn
        import numpy as np
        tf.config.set_visible_devices([], "GPU")
        tf.config.experimental.enable_op_determinism()
        from data import load_data
        from training import run_experiment
        from visualization import save_plots
        metadata = dict(config=asdict(config), python=platform.python_version(),
                        tensorflow=tf.__version__, keras=tf.keras.__version__,
                        numpy=np.__version__, sklearn=sklearn.__version__,
                        platform=platform.platform(), device="CPU", status="running")
        metadata_path = output / "run.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logging.info("Загрузка IMDB; результаты: %s", output)
        data = load_data(config)
        np.savez(output / "split_ids.npz", initial=data.initial_ids, pool=data.pool_ids,
                 validation=data.validation_ids, test=data.test_ids)
        validation, test = run_experiment(config, data, output)
        save_plots(validation, test, output)
        metadata["status"] = "complete"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for row in test:
            logging.info("TEST %s: accuracy=%.4f F1=%.4f FP=%d FN=%d",
                         row["strategy"], row["accuracy"], row["f1"],
                         row["false_positive"], row["false_negative"])
        logging.info("Готово: %s", output.resolve())
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        logging.error("Эксперимент не завершён: %s", error)
        return 1
    except Exception:
        logging.exception("Неожиданная ошибка; см. traceback")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
