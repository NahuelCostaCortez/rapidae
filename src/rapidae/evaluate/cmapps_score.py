import numpy as np

from rapidae.evaluate.metric import Metric


class CMAPSS_Score(Metric):
    """
    CMAPSS Score Metric for the PHM 2008 Challenge.

    This metric is specifically designed for the C-MAPSS dataset in the Prognostics Health Management (PHM) 2008 Challenge by NASA.
    It computes a weighted sum of Remaining Useful Life (RUL) errors, where the scoring function is an asymmetric function that penalizes
    late predictions more than early predictions.

    The scoring function is defined as follows:
    - If the predicted RUL is less than the true RUL, penalize with exp(-subs/10) - 1.
    - If the predicted RUL is greater than or equal to the true RUL, penalize with exp(subs/13) - 1.

    The metric is designed to encourage models to make earlier and more accurate predictions of remaining useful life.

    For more information on the C-MAPSS dataset and the PHM 2008 Challenge, see: https://data.nasa.gov/Raw-Data/PHM-2008-Challenge/nk8v-ckry

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
                results = results + np.exp(-subs / 10)[0] - 1
            else:
                results = results + np.exp(subs / 13)[0] - 1

        return np.array(results)
