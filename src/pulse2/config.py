import logging
from logging import config as logging_config

import dolfin

comm = dolfin.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0
fancy_format = (
    "%(asctime)s %(rank)s(%(filename)s:%(lineno)d) - %(levelname)s - " "%(message)s"
)
base_format = "%(asctime)s (%(filename)s:%(lineno)d) - %(levelname)s - %(message)s"
root_format = "ROOT -" + base_format


class FancyFormatter(logging.Formatter):
    "Custom Formatter for logging"

    def format(self, record: logging.LogRecord) -> str:
        # Set the rank attribute with is a parameter in the `fancy_format``
        record.rank = f"CPU {rank}: " if size > 1 else ""

        # Create the formatter
        formatter = logging.Formatter(fancy_format)
        # Format the record
        return formatter.format(record)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "root": {"format": root_format},
        "base": {"format": base_format},
        "default": {"()": FancyFormatter},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
        "root_console": {"class": "logging.StreamHandler", "formatter": "root"},
    },
    "loggers": {
        "pulse2": {
            "handlers": ["console"],
            "level": "INFO",
            # Don't send it up my namespace for additional handling
            "propagate": False,
        },
        "FFC": {"level": "WARNING"},
        "UFL": {"level": "WARNING"},
        "UFL_LEGACY": {"level": "WARNING"},
        "dolfin": {"level": "INFO"},
        "dijitso": {"level": "INFO"},
    },
    "root": {"handlers": ["root_console"], "level": "INFO"},
}


logging_config.dictConfig(LOGGING_CONFIG)
