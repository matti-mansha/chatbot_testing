# logging_config.py
"""
Centralized logging configuration for the testing system.
Sets up logging to both console and file with rotation.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def setup_logging(
    service_name: str,
    log_level: str = None,
    log_dir: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging for a service with both file and console handlers.
    
    Args:
        service_name: Name of the service (e.g., 'test_bot', 'test_execution')
        log_level: Log level from env or default to INFO
        log_dir: Directory for log files, defaults to ./logs
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Get log directory from environment or parameter
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", "logs")
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation
    log_file = log_path / f"{service_name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # File gets everything
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log the initialization
    logger.info(f"=" * 80)
    logger.info(f"Logging initialized for {service_name}")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"=" * 80)
    
    return logger


def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context about where the exception occurred
    """
    import traceback
    
    if context:
        logger.error(f"Exception in {context}: {type(exc).__name__}: {exc}")
    else:
        logger.error(f"Exception: {type(exc).__name__}: {exc}")
    
    logger.debug("Full traceback:", exc_info=True)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with its parameters.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Parameters being passed to the function
    """
    params = ", ".join([f"{k}={repr(v)[:100]}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")


def log_api_call(logger: logging.Logger, method: str, url: str, status_code: int = None, duration: float = None):
    """
    Log an API call.
    
    Args:
        logger: Logger instance
        method: HTTP method (GET, POST, etc.)
        url: URL being called
        status_code: Response status code (if available)
        duration: Request duration in seconds (if available)
    """
    msg = f"{method} {url}"
    
    if status_code is not None:
        msg += f" → {status_code}"
    
    if duration is not None:
        msg += f" ({duration:.2f}s)"
    
    if status_code and status_code >= 400:
        logger.warning(f"API call failed: {msg}")
    else:
        logger.debug(f"API call: {msg}")


# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    logger = setup_logging("test_service")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log_exception(logger, e, "test context")
    
    log_function_call(logger, "test_function", param1="value1", param2=123)
    log_api_call(logger, "POST", "https://api.example.com/test", 200, 1.5)
    log_api_call(logger, "GET", "https://api.example.com/error", 404, 0.5)