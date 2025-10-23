# src/logging_config.py
from loguru import logger
import sys

# Avoid adding handlers multiple times if imported repeatedly
logger.remove()

log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Console handler (Docker logs)
logger.add(sys.stdout, format=log_format, level="INFO", colorize=True, enqueue=True)

# File handler (mounted in docker-compose)
logger.add(
    "logging.log",
    format=log_format,
    level="DEBUG",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# Export the logger for other modules
__all__ = ["logger"]