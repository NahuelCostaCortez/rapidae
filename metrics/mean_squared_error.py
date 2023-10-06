from sklearn.metrics import mean_squared_error

from .metric import Metric


class MeanSquaredError(Metric):
    """
    Wrapper class of the scikit-learn function mean_squared_error.
    """

    def __init__(self, squared=True):
        super().__init__()
        self.squared = squared

    def calculate(self, y_true, y_hat):
        """
        Calculates the metric for the model's results.

        Args
        ----
            y_true (ArrayLike): Ground truth (correct) target values.
            y_hat (ArrayLike): Estimated target values.
        """
        return mean_squared_error(y_true, y_hat, squared=self.squared)
