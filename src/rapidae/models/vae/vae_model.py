from typing import Tuple, Union
from keras import metrics, ops, losses
from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class VAE(BaseAE):
    """
    Variational Autoencoder (VAE) model.

    Args:
        encoder (BaseEncoder): An instance of BaseEncoder.
        decoder (BaseDecoder): An instance of BaseDecoder.
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        exclude_decoder (bool): Flag to exclude the decoder.
        downstream_task (str): Downstream task, can be 'regression' or 'classification'.
        layers_conf (list): List specifying the configuration of layers for custom models.
        recon_loss_fn (str): Whether to use "mse" or "bce". Default is set to None, "mse" or "bce" will be chosen depending on the data dimensions.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        exclude_decoder: bool = False,
        downstream_task: str = None,
        recon_loss_fn: str = None,
        **kwargs,
    ):
        # Initialize base class
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            decoder=decoder,
        )

        # Initialize rest of attributes
        self.exclude_decoder = exclude_decoder
        self.downstream_task = downstream_task
        self.recon_loss_fn = recon_loss_fn

        self.sampling = Normal()
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

        # Check whether a downstream task has been selected
        if self.downstream_task is not None:
            self.check_downstream(downstream_task, kwargs)
        else:
            self.logger.log_info("No specific downstream task has been selected")

        # Check whether to keep the decoder
        if not self.exclude_decoder:
            self.check_recon_loss()
            self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")


    def check_downstream(self, downstream_task, kwargs):
        self.downstream_task = downstream_task.lower()

        if self.downstream_task == "regression":
            from rapidae.models.base import BaseRegressor

            self.logger.log_info("Regressor set for the latent space")
            self.regressor = BaseRegressor()
            self.reg_loss_tracker = metrics.Mean(name="reg_loss")

        elif self.downstream_task == "classification":
            from rapidae.models.base import BaseClassifier

            self.logger.log_info("Classifier set for the latent space")
            self.classifier = BaseClassifier(kwargs["n_classes"])
            self.weight_clf = (
                kwargs["weight_clf"] if "weight_clf" in kwargs else 1.0
            )
            self.clf_loss_tracker = metrics.Mean(name="clf_loss")

        else:
            self.logger.log_warning(
                'The downstream task is not a valid string. Available options are "regression" and "classification"'
            )

    def check_recon_loss(self):
        if self.recon_loss_fn == "mse" or len(self.input_dim)==2:
            self.recon_loss_fn = losses.mean_squared_error
        elif self.recon_loss_fn == "bce" or len(self.input_dim)==3:
            self.recon_loss_fn = losses.binary_crossentropy
        else:
            self.recon_loss_fn = losses.mean_squared_error
        self.logger.log_info("Using " + str(self.recon_loss_fn.__name__) + " as the reconstruction loss function")

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
        if not self.exclude_decoder:
            x_recon = self.decoder(z)
            outputs["x_recon"] = x_recon

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
        loss = kl_loss

        # Reconstruction loss
        if not self.exclude_decoder:
            recon_loss = self.recon_loss_fn(x, y_pred["x_recon"])
            recon_loss = ops.mean(
                ops.sum(recon_loss, axis=tuple(range(1, recon_loss.ndim))) # (1,2) in case of images, 1 in case of time_series
            )
            self.reconstruction_loss_tracker.update_state(recon_loss)
            loss += recon_loss

        # Downstream loss
        if self.downstream_task is not None:
            # Regressor loss
            if self.downstream_task == "regression":
                reg_loss = ops.mean(losses.mean_squared_error(y, y_pred["reg"]))
                self.reg_loss_tracker.update_state(reg_loss)
                loss += reg_loss

            # Classifier loss
            if self.downstream_task == "classification":
                clf_loss = ops.mean(losses.categorical_crossentropy(y, y_pred["clf"]))
                self.clf_loss_tracker.update_state(clf_loss)
                loss += self.weight_clf * clf_loss

        return loss
