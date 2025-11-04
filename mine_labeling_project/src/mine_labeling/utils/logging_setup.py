"""
Logging Setup Utility
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(config: dict = None,
                  log_level: str = "INFO",
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        config: Configuration dictionary
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path (optional)

    Returns:
        Configured logger instance
    """
    # Get logging configuration
    if config and 'logging' in config:
        logging_config = config['logging']
        log_level = logging_config.get('level', log_level)
        log_file = logging_config.get('log_file', log_file)
    else:
        logging_config = {}

    # Create logger
    logger = logging.getLogger('mine_labeling')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    if logging_config.get('console_enabled', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))

        console_format = logging_config.get(
            'console_format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)

    # File handler
    if logging_config.get('file_enabled', True) and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if logging_config.get('rotate_logs', True):
            # Rotating file handler
            max_bytes = logging_config.get('max_bytes', 10 * 1024 * 1024)  # 10MB
            backup_count = logging_config.get('backup_count', 5)

            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            # Standard file handler
            file_handler = logging.FileHandler(log_path)

        file_handler.setLevel(getattr(logging, log_level.upper()))

        file_format = logging_config.get(
            'file_format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'mine_labeling') -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
