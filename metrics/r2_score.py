from sklearn.metrics import r2_score

from .metric import Metric


class R2Score(Metric):
    """
    Wrapper class of the scikit-learn function r2_score.
    """

    def __init__(self, force_finite=True):
        super().__init__()
        self.force_finite = force_finite

    def calculate(self, y_true, y_hat):
        """
        Calculates the metric for the model's results.

        Args
        ----
            y_true (ArrayLike): Ground truth (correct) target values.
            y_hat (ArrayLike): Estimated target values.
        """
        return r2_score(y_true, y_hat, force_finite=self.force_finite)
