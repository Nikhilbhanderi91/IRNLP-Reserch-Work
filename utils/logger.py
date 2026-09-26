"""
================================================
utils/logger.py
Centralised logging configuration for the project
================================================
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger that writes to stdout and to a rotating log file.

    Args:
        name:  Module/class name (use __name__).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # ── Console handler ──────────────────────────────
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        # ── File handler ─────────────────────────────────
        fh = logging.FileHandler(log_dir / "hmrp.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
