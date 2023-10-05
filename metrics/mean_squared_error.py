from sklearn.metrics import mean_squared_error

from .metric import Metric


class MeanSquaredError(Metric):
    """
    Wrapper class of the sklearn mean_squared_error
    """

    def __init__(self, squared=True):
        """
        Squared parameter determine if the desired metric is the Mean Squared Error (squared=True)
        or the Root Mean Squared Error (squared=False)
        """
        super().__init__()
        self.squared = squared

    def calculate(self, y_true, y_hat):
        return mean_squared_error(y_true, y_hat, squared=self.squared)
        