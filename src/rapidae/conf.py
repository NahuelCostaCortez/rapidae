import logging
import os

from colorlog import ColoredFormatter


class Logger:
    """
    Class to manage logs in the library.
    """

    _instance = None

    def __new__(
        cls,
        log_file=os.path.join(
            os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, "rapidae.log"
        ),
    ):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger(log_file=log_file)
            cls._instance.logger.setLevel(logging.INFO) # Needed for Colab, otherwise it will display only warnings
        return cls._instance

    def _initialize_logger(self, log_file="rapidae.log"):
        # Advanced configuration (colors and format) of the loggar
        formatter = ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)s]%(reset)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Basic configuration of the logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_file,
        )

        self.logger = logging.getLogger("RAPIDAE_logger")

        # Display logs on console
        console_handler = logging.StreamHandler()

        # Add formatter
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.propagate=False # Avoid duplicate logs

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
    # Usage example: Testing purposes
    rapidae_logger1 = Logger()
    rapidae_logger1.log_info("Testing log info ...")

    rapidae_logger2 = Logger()
    rapidae_logger2.log_warning("Testing log warning ...")

    rapidae_logger3 = Logger()
    rapidae_logger3.log_error("Testing log error ...")

    rapidae_logger4 = Logger()
    rapidae_logger4.log_debug("Testing log debug ...")
