from typing import Tuple, Union

import keras
import torch

from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class TimeVAE(BaseAE):
    """
    A Variational Autoencoder (VAE)  for multivariate time series generation.

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
        beta: float = 3.0,
        min_std: float = 0.25,
    ):
        # Initialize base class
        BaseAE.__init__(self, input_dim, latent_dim, encoder, decoder)

        # Parameters for distribution
        self.beta = beta
        self.min_std = min_std

        # Training metrics trackers
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    def call(self, x):
        # ENCODER
        mu, log_var = self.encoder(x)

        # sample z from q - encoder
        std = keras.ops.exp(0.5 * log_var)  # convert log_var to std
        q = Normal(mu, std)
        z = q.sample()

        # DECODER
        recon_x, recon_x_log_scale = self.decoder(z)

        outputs = {
            "z": z,
            "z_mean": mu,
            "z_std": std,
            "x_recon": recon_x,
            "x_log_scale": recon_x_log_scale,
        }

        return outputs

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = keras.ops.exp(logscale) + self.min_std
        dist = Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return keras.ops.sum(log_pxz, axis=(1, 2))

    """
    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale) + self.min_std
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))
    """

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

    """
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl
    """

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(
            y_pred["x_recon"], y_pred["x_log_scale"], x
        )
        self.reconstruction_loss_tracker.update_state(recon_loss)

        # kl
        kl = self.kl_divergence(y_pred["z"], y_pred["z_mean"], y_pred["z_std"])
        self.kl_loss_tracker.update_state(kl)

        # elbo
        elbo = kl - (self.beta * recon_loss)
        return keras.ops.mean(elbo)
