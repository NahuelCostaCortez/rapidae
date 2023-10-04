from .metric import Metric
from sklearn.metrics import accuracy_score

class AccuracyScore(Metric):
    """
    Wrapper class of the sklearn accuracy_scores
    """

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def calculate(self, y_true, y_hat):
        return accuracy_score(y_true, y_hat, normalize=self.normalize)