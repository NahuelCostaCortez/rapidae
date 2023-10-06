from sklearn.metrics import accuracy_score

from .metric import Metric


class AccuracyScore(Metric):
    """
    Wrapper class of the scikit-learn function accuracy_score.
    """

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def calculate(self, y_true, y_hat):
        """
        Calculates the metric for the model's results.

        Args
        ----
            y_true (ArrayLike): Ground truth (correct) labels.
            y_hat (ArrayLike): Predicted labels, as returned by a classifier.
        """
        return accuracy_score(y_true, y_hat, normalize=self.normalize)
