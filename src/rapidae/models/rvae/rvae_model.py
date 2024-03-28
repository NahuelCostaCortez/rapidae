from typing import Tuple, Union

from keras import metrics, ops, losses
from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class RVAE(BaseAE):
    """
    Recurrent Variational AutoEncoder (RVAE) model - DOI: 10.1109/ACCESS.2021.3064854

    Args:
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        encoder (BaseEncoder): An instance of BaseEncoder.
        decoder (BaseDecoder): An instance of BaseDecoder.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
    ):
        # Initialize base class
        BaseAE.__init__(self, input_dim, latent_dim, encoder, decoder)

        # Initialize sampling distribution
        self.sampling = Normal()

        # Initialize metrics
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")

    # keras model call function
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
        recon_loss = losses.mean_squared_error(x, y_pred["x_recon"])
        recon_loss = ops.mean(recon_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)

        loss = kl_loss + recon_loss

        return loss
