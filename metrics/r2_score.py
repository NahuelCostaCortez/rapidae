from sklearn.metrics import r2_score

from .metric import Metric


class R2Score(Metric):
    """
    Wrapper class of the sklearn R2Score.
    """

    def __init__(self, force_finite=True):
        super().__init__()
        self.force_finite = force_finite

    def calculate(self, y_true, y_hat):
        return r2_score(y_true, y_hat, force_finite=self.force_finite)