import numpy as np

from rapidae.metrics.metric import Metric


class CMAPSS_Score(Metric):
    """
    Specific metric designed for CMAPPS dataset in the PHM 2008 Challenge by NASA.
    This score is a weighted sum of RUL errors where the scoring function is an asymmetric function
    that penalizes late predictions more than the early predictions.
    See more info at https://data.nasa.gov/Raw-Data/PHM-2008-Challenge/nk8v-ckry
    """

    def __init__(self):
        super().__init__()

    def calculate(self, y_true, y_hat):
        """
        Calculates the metric for the model's results.

        Args
        ----
            y_true (ArrayLike): Ground truth (correct) labels.
            y_hat (ArrayLike): Predicted RUL.
        """
        results = 0

        for true, hat in zip(y_true, y_hat):
            subs = hat - true
            if subs < 0:
                results = results + np.exp(-subs/10)[0]-1
            else:
                results = results + np.exp(subs/13)[0]-1
        return np.array(results)
