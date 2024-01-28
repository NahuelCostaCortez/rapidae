from abc import ABC, abstractmethod

from rapidae.conf import Logger


class Metric(ABC):
    """
    Base class for all metrics used in the library.

    Usage:
        # Example usage of a custom metric class
        class CustomMetric(Metric):
            def calculate(self, y_true, y_hat):
                # Custom metric calculation logic
                pass
        custom_metric = CustomMetric()
        custom_metric.show_start_message()
        result = custom_metric.calculate(y_true, y_hat)
    """

    @abstractmethod
    def calculate(self, y_true, y_hat):
        """
        Abstract method to calculate the metric for the model's results.

        Args:
            y_true (ArrayLike): Ground truth (correct) labels.
            y_hat (ArrayLike): Predicted values.
        """
        pass


    def show_start_message(self):
        """
        Prints the name of the selected metric in the terminal for logging purposes.
        """
        self.logger = Logger()
        self.logger.log_info(
            '+++ Metric {} +++'.format(self.__class__.__name__))
