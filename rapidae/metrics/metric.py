from abc import ABC, abstractmethod

from rapidae.conf import Logger


class Metric(ABC):
    """
    Base class for all metrics used in the library.
    """

    @abstractmethod
    def calculate(self, y_true, y_hat):
        """
        Calculates the metric for the model's results.
        """
        pass


    def show_start_message(self):
        """
        Prints name of the selected metric in the terminal. Logging purposes
        """
        self.logger = Logger()
        self.logger.log_info(
            '+++ Metric {} +++'.format(self.__class__.__name__))
