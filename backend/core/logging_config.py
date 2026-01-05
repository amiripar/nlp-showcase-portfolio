import logging
import os


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """
    Minimal logging configuration:
    - Logs to terminal
    - Logs to logs/app.log
    """
  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),             
            logging.FileHandler(log_file, encoding="utf-8"),  
        ],
    )
