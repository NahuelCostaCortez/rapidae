from typing import Tuple, Union
from keras import metrics, ops, losses
from rapidae.conf import Logger
from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class Beta_VAE(BaseAE):
    """
    Variational Autoencoder (VAE) model.

    Args:
        encoder (BaseEncoder): An instance of BaseEncoder.
        decoder (BaseDecoder): An instance of BaseDecoder.
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        layers_conf (list): List specifying the configuration of layers for custom models.
        **kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        beta: float = 1.0,
        recon_loss_fn: str = None,
    ):
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            decoder=decoder,
        )

        if beta < 0.0:
            self.logger.log_warning(
                "The beta parameter should not be a value less than or equal to zero."
            )
        else:
            self.beta = beta

        self.sampling = Normal()
        self.recon_loss_fn = recon_loss_fn
        self.check_recon_loss()

        # Training metrics trackers
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

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
        x_recon = self.decoder(z)

        return {"z": z, "z_mean": z_mean, "z_log_var": z_log_var, "x_recon": x_recon}

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

        # Reconstruction loss
        recon_loss = self.recon_loss_fn(x, y_pred["x_recon"])
        recon_loss = ops.mean(
            ops.sum(recon_loss, axis=tuple(range(1, recon_loss.ndim))) # (1,2) in case of images, 1 in case of time_series
        )
        self.reconstruction_loss_tracker.update_state(recon_loss)

        loss = (self.beta * kl_loss) + recon_loss

        return loss
