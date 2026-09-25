"""
Default log output for the terrain package.

Every module logs through ``logging.getLogger(__name__)``, i.e. children of
the ``src.terrain`` package logger. This installs one fallback console
handler on that package logger which prints INFO and above *only while the
application has not configured logging itself* (root logger has no
handlers). Once a script calls ``logging.basicConfig`` or adds its own
handlers, messages flow only through those, so nothing is printed twice.
"""

import logging

PACKAGE_LOGGER = logging.getLogger(__name__.rpartition(".")[0])
_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class _UnconfiguredAppFilter(logging.Filter):
    """Pass records only when the root logger has no handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not logging.getLogger().handlers


def install_default_handler() -> None:
    """Idempotently attach the fallback handler to the package logger."""
    if any(getattr(h, "_terrain_default", False) for h in PACKAGE_LOGGER.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_UnconfiguredAppFilter())
    handler._terrain_default = True
    PACKAGE_LOGGER.addHandler(handler)
    if PACKAGE_LOGGER.level == logging.NOTSET:
        PACKAGE_LOGGER.setLevel(logging.INFO)
