"""Logging configuration for AI Voice Detection System"""

import logging
import logging.config
import os
from typing import Dict, Any

from pythonjsonlogger import jsonlogger


def get_logging_config() -> Dict[str, Any]:
    """
    Returns logging configuration dictionary.
    
    Returns:
        Dictionary containing logging configuration
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json")  # json or text
    
    if log_format == "json":
        formatter_class = "pythonjsonlogger.jsonlogger.JsonFormatter"
        format_string = "%(asctime)s %(name)s %(levelname)s %(message)s"
    else:
        formatter_class = "logging.Formatter"
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "class": formatter_class,
                "format": format_string,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "default",
                "filename": "logs/ai_voice_detector.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "ai_voice_detector": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }
    
    return config


def setup_logging() -> None:
    """
    Sets up logging configuration for the application.
    Creates logs directory if it doesn't exist.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Apply logging configuration
    config = get_logging_config()
    logging.config.dictConfig(config)
    
    # Get logger and log startup message
    logger = logging.getLogger("ai_voice_detector")
    logger.info("Logging configuration initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Gets a logger instance with the specified name.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"ai_voice_detector.{name}")