from typing import Tuple, Union

import keras

from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class TimeHVAE(BaseAE):
    """
    Hierarchical Variational Autoencoder (VAE) for multivariate time series.

    Args:
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        encoder (BaseEncoder): An instance of BaseEncoder.
        decoder (BaseDecoder): An instance of BaseDecoder.
        enc_out_dim (int): Output dimension of the encoder.
        beta (float): Beta value for the KL divergence loss.
        min_std (float): Minimum standard deviation value.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 16,
        encoder: callable = None,
        decoder: callable = None,
        nz: int = 1,
        beta: float = 3.0,
        min_std: float = 0.25,
    ):
        # Initialize base class
        BaseAE.__init__(self, input_dim, latent_dim, encoder, decoder)

        # level of hierarchy
        self.nz = nz
        self.depth = nz - 1  # number of layers in the hierarchy

        # Parameters for distribution
        self.beta = beta
        self.min_std = min_std

        # Training metrics trackers
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    def call(self, x):

        # log_enc = []
        # log_dec = []
        kls = []
        recon_losses = []

        for i in range(self.nz):
            # ENCODER
            # get the parameters of inference distribution i given x q(z_i|x) or z q(z_i|z_{i+1})
            mu, log_var = self.encoder(x, lvl=i)

            # sample z from q - encoder
            std = keras.ops.exp(0.5 * log_var)  # convert log_var to std
            q = Normal(mu, std)
            z_next = q.sample()

            # kl - likelihood of z
            kl = self.kl_divergence(z_next, mu, std)
            kls = kl

            # DECODER
            # get the parameters of generative distribution i p(x_i|z_i) or z p(z_i|z_{i+1})
            x_recon, x_log_scale = self.decoder(z_next, lvl=i)

            # reconstruction loss
            recon_loss = self.gaussian_likelihood(x_recon, x_log_scale, x)
            recon_losses = recon_loss

        """
        outputs = {
            "z": z_next,
            "z_mean": mu,
            "z_std": std,
            "x_recon": x_recon,
            "x_log_scale": x_log_scale,
        }
        """

        return {"x_recon": x_recon, "kl": kls, "recon_loss": recon_losses}

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = keras.ops.exp(logscale) + self.min_std
        dist = Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return keras.ops.sum(log_pxz, axis=(1, 2))

    def kl_divergence(self, z, mu, std):
        p = Normal(
            keras.ops.zeros_like(mu), keras.ops.ones_like(std)
        )  # prior - standard normal
        q = Normal(mu, std)  # posterior - encoder - learned distribution

        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = keras.ops.sum(kl, axis=-1)
        return kl

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(
            y_pred["x_recon"], y_pred["x_log_scale"], x
        )
        self.reconstruction_loss_tracker.update_state(recon_loss)

        # kl
        kl = self.kl_divergence(y_pred["z"], y_pred["z_mean"], y_pred["z_std"])
        self.kl_loss_tracker.update_state(kl)
        """
        # reconstruction loss
        recon_loss = y_pred["recon_loss"]
        self.reconstruction_loss_tracker.update_state(recon_loss)

        # kl
        kl = y_pred["kl"]
        self.kl_loss_tracker.update_state(kl)

        # elbo
        elbo = kl - (self.beta * recon_loss)
        return keras.ops.mean(elbo)
