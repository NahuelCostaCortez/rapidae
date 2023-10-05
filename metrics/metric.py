from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Base class for all metrics used in the library.
    """

    @abstractmethod
    def calculate(self, y_true, y_hat):
        """
        Calculate the corresponding metric on the results of a model
        """
        pass

    def show_start_message(self):
        """
        Print name of the selected metric in the terminal. Logging purposes
        """
        print("\n+++ Metric {} +++".format(self.__class__.__name__))


