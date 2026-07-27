"""Centralized logging utilities for the ExplainableModel project.

Provides a single :func:`get_logger` factory that returns a configured
``logging.Logger`` instance. The logger writes to both the console and an
optional rotating file, with a consistent format across the codebase.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILE = "app.log"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_BACKUP_COUNT = 3


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Set the root logger level and ensure it has at least one handler.

    Safe to call multiple times: handlers are only added when missing.

    Args:
        level: Minimum logging level for the root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(handler)


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_dir: str = _DEFAULT_LOG_DIR,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    The logger propagates to the root logger (which has a console handler),
    and optionally writes to a rotating file under ``log_dir``.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_file: Optional file name created under ``log_dir``. When ``None``
            only console output is produced.
        level: Logging level for this logger.
        log_dir: Directory where the log file is created when ``log_file`` is set.
        max_bytes: Maximum size of the log file before rotation.
        backup_count: Number of rotated backup files to keep.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    _configure_root_logger(level=level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True

    if log_file is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(file_handler)

    return logger


__all__ = ["get_logger"]