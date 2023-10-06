from abc import ABC, abstractmethod


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
        print("\n+++ Metric {} +++".format(self.__class__.__name__))
