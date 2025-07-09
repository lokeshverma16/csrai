"""
Logging Configuration

Centralized logging setup for the Customer Analytics & Recommendation System
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import settings
from .settings import LOGS_DIR, APP_NAME

def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Set up centralized logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_filename: Custom log filename (optional)
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"pipeline_{timestamp}.log"
        
        log_file_path = LOGS_DIR / log_filename
        
        # Create rotating file handler (10MB max, 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the centralized configuration
    
    Args:
        name: Logger name (optional, defaults to APP_NAME)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = APP_NAME
    
    return logging.getLogger(name) 