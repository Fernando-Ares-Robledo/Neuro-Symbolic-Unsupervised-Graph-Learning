import os
import logging
from datetime import datetime
from logging.config import dictConfig

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILENAME = os.path.join(LOG_DIR, f"Log_{current_time}.log")

class NoSilentFilter(logging.Filter):
    def filter(self, record):
        return not getattr(record, 'silent', False)

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'no_silent': {
            '()': NoSilentFilter,
        }
    },
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOG_FILENAME,
            'formatter': 'standard',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'filters': ['no_silent'],  
        }
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },
}

dictConfig(logging_config)

# This message will be both printed and saved (using logging.info normally)
#logging.info("This log message is visible in console and saved in log file.")

# This message is "silent" on the console and will only be saved (using extra attribute 'silent')
# logging.info("This log message will be logged silently to the file only.", extra={'silent': True})
