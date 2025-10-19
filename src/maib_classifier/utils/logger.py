"""
Logger configuration for MAIB Incident Type Classifier.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
  name: str = "maib_classifier",
  level: str = "INFO",
  log_file: Optional[str] = None,
  format_string: Optional[str] = None
) -> logging.Logger:
  """
  Setup logger with console and optional file output.

  Args:
    name: Logger name
    level: Logging level
    log_file: Optional log file path
    format_string: Optional custom format string

  Returns:
    Configured logger instance
  """
  logger = logging.getLogger(name)
  logger.setLevel(getattr(logging, level.upper()))

  # Clear existing handlers
  logger.handlers.clear()

  # Default format
  if format_string is None:
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  formatter = logging.Formatter(format_string)

  # Console handler
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  # File handler (if specified)
  if log_file:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

  return logger


def get_logger(name: str = "maib_classifier") -> logging.Logger:
  """Get existing logger or create new one."""
  return logging.getLogger(name)
