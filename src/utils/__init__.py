# Convenience re-exports for the training pipeline (Hydra + Lightning).
# Each import is wrapped in try/except so that loading the model through
# torch.hub does not require hydra/lightning/rich to be installed. The
# training entry points (src/train.py, src/eval.py, ...) declare those deps.

try:
    from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
except ImportError:  # hub-only install path
    instantiate_callbacks = instantiate_loggers = None  # type: ignore

try:
    from src.utils.logging_utils import log_hyperparameters
except ImportError:
    log_hyperparameters = None  # type: ignore

try:
    from src.utils.pylogger import get_pylogger
except ImportError:
    import logging

    def get_pylogger(name: str = __name__):  # type: ignore[override]
        """Lightweight fallback when rich/hydra are not installed."""
        return logging.getLogger(name)

try:
    from src.utils.rich_utils import enforce_tags, print_config_tree
except ImportError:
    enforce_tags = print_config_tree = None  # type: ignore

try:
    from src.utils.utils import extras, get_metric_value, task_wrapper
except ImportError:
    extras = get_metric_value = task_wrapper = None  # type: ignore
