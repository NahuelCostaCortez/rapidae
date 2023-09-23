from .metric import Metric
from sklearn.metrics import r2_score

class R2Score(Metric):

    def __init__(self, force_finite=True):
        super().__init__()
        self.force_finite = force_finite

    def calculate(self, y_true, y_hat):
        self.show_start_message()
        return r2_score(y_true, y_hat, force_finite=self.force_finite)