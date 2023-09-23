from .metric import Metric
from sklearn.metrics import mean_squared_error

class MeanSquaredError(Metric):

    def __init__(self, squared=False):
        super().__init__()
        self.squared = squared

    def calculate(self, y_true, y_hat):
        self.show_start_message()
        return mean_squared_error(y_true, y_hat, squared=self.squared)
        