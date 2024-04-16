from typing import Tuple, Union
from keras import metrics, ops, losses
from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class RVE(BaseAE):
    """
    Recurrent Variational Encoder (RVE) model.

    Args:
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        encoder (BaseEncoder): An instance of BaseEncoder.
        downstream_task (str): Downstream task, can be 'regression' or 'classification'.
        **kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        downstream_task: str = "regression",
        **kwargs,
    ):
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            **kwargs,
        )

        self.sampling = Normal()
        self.downstream_task = downstream_task.lower()

        if self.downstream_task == "regression":
            from rapidae.models.base import BaseRegressor

            self.regressor = BaseRegressor()
            self.logger.log_info("Regressor set for the latent space.")
            self.reg_loss_tracker = metrics.Mean(name="reg_loss")

        elif self.downstream_task == "classification":
            from rapidae.models.base import BaseClassifier

            self.logger.log_info("Setting classifier for the latent space...")
            self.classifier = BaseClassifier(kwargs["n_classes"])
            self.weight_vae = kwargs["weight_vae"] if "weight_vae" in kwargs else 1.0
            self.weight_clf = kwargs["weight_clf"] if "weight_clf" in kwargs else 1.0
            self.clf_loss_tracker = metrics.Mean(name="clf_loss")

        else:
            self.logger.log_warning(
                'The downstream task is not a valid string. Available options are "regression" and "classification"'
            )

        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def call(self, x):
        z_mean, z_log_var = self.encoder(x)
        z_std = ops.exp(0.5 * z_log_var)
        z = self.sampling([z_mean, z_std])
        outputs = {}
        outputs["z"] = z
        outputs["z_mean"] = z_mean
        outputs["z_log_var"] = z_log_var
        if self.downstream_task == "regression":
            reg_prediction = self.regressor(z)
            outputs["reg"] = reg_prediction
        if self.downstream_task == "classification":
            clf_prediction = self.classifier(z)
            outputs["clf"] = clf_prediction

        return outputs

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # KL loss
        kl_loss = -0.5 * (
            1
            + y_pred["z_log_var"]
            - ops.square(y_pred["z_mean"])
            - ops.exp(y_pred["z_log_var"])
        )
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        self.kl_loss_tracker.update_state(kl_loss)

        # Regressor loss
        if self.downstream_task == "regression":
            reg_loss = ops.mean(losses.mean_squared_error(y, y_pred["reg"]))
            self.reg_loss_tracker.update_state(reg_loss)
            loss = kl_loss + reg_loss

        # Classifier loss
        if self.downstream_task == "classification":
            clf_loss = ops.mean(
                losses.categorical_crossentropy(y, y_pred["clf"])
            )
            self.clf_loss_tracker.update_state(clf_loss)
            loss = self.weight_vae * kl_loss + self.weight_clf * clf_loss

        return loss
