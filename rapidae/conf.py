import logging
import os
from os import path

from colorlog import ColoredFormatter

# TODO: Eliminar
class Directory:
    DATA = path.join('.', 'data')
    OUTPUT_DIR = path.join('.', 'output_dir')


class RandomSeed:
    RANDOM_SEED = 1


class Logger:
    """
    Class to manage logs in the library.
    """
    _instance = None

    def __new__(cls, log_file=os.path.join('..', 'rapidae.log')):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger(log_file=log_file)
        return cls._instance

    def _initialize_logger(self, log_file='rapidae.log'):
        # Create directory if needed
        #if not path.exists(Directory.BASE):
        #    makedirs(Directory.BASE)

        # Advanced configuration (colors and format) of the loggar
        formatter = ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)s]%(reset)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
            }
        )

        # Basic configuration of the logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_file
        )

        self.logger = logging.getLogger("RAPIDAE_logger")

        # Display logs on console
        console_handler = logging.StreamHandler()

        # Add formatter
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_info(self, message):
        """Register info message"""
        self.logger.info(message)

    def log_warning(self, message):
        """Register warning message"""
        self.logger.warning(message)

    def log_error(self, message):
        """Register error message"""
        self.logger.error(message)

    def log_debug(self, message):
        """Register debug message"""
        self.logger.debug(message)


if __name__ == "__main__":

    # Usage example
    aepy_logger1 = Logger()
    aepy_logger1.log_info('Testing log info ...')

    aepy_logger2 = Logger()
    aepy_logger2.log_warning('Testing log warning ...')

    aepy_logger3 = Logger()
    aepy_logger3.log_error('Testing log error ...')

    aepy_logger4 = Logger()
    aepy_logger4.log_debug('Testing log debug ...')
