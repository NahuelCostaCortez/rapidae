from typing import Tuple, Union

from keras import metrics, ops

from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class TimeVAE(BaseAE):
    """
    A Variational Autoencoder (VAE)  for multivariate time series generation.
    Contrary to vanilla VAE whose decoder output is the reconstruction of the data, 
    the decoder outputs the mean and log var of the input data.

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
        # for encoder
        self.normal = Normal()

        # Training metrics trackers
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")

    def call(self, x):
        # ENCODER
        z_mean, z_log_var = self.encoder(x)

        # sample z from q - encoder
        z_std = ops.exp(0.5 * z_log_var)  # convert log_var to std
        z = self.normal([z_mean, z_std])

        # DECODER
        x_mean, x_log_var = self.decoder(z)  # mu is used as the reconstruction

        outputs = {
            "z": z,
            "z_mean": z_mean,
            "z_std": z_std,
            "x_mean": x_mean,
            "x_log_var": x_log_var,
        }

        return outputs

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = ops.exp(logscale) + self.min_std
        log_pxz= self.normal.log_prob([sample, mean, scale])
        return ops.sum(log_pxz, axis=(1, 2))

    """
    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale) + self.min_std
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))
    """

    def kl_divergence(self, z, mu, std):
        log_pz = self.normal.log_prob([z, ops.zeros_like(mu), ops.ones_like(std)])
        log_qzx = self.normal.log_prob([z, mu, std])

        # kl
        kl = log_qzx - log_pz
        kl = ops.sum(kl, axis=-1)
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
            y_pred["x_mean"], y_pred["x_log_var"], x
        )
        self.reconstruction_loss_tracker.update_state(recon_loss)

        # kl loss
        kl = self.kl_divergence(y_pred["z"], y_pred["z_mean"], y_pred["z_std"])
        self.kl_loss_tracker.update_state(kl)

        # elbo
        elbo = kl - (self.beta * recon_loss)
        return ops.mean(elbo)
    
    def reconstruct(self, x):
        outputs = self.call(x)
        x_mean, x_log_var = outputs["x_mean"], outputs["x_log_var"]
        x_std = ops.exp(0.5 * x_log_var)
        return self.normal([x_mean, x_std])
