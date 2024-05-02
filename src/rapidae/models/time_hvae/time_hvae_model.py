from typing import Tuple, Union

from keras import metrics, ops

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
        self.normal = Normal()

        # Training metrics trackers
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = ops.exp(logscale) + self.min_std
        log_pxz = self.normal.log_prob([sample, mean, scale])
        #return ops.sum(log_pxz, axis=(1, 2))
        return ops.sum(log_pxz, axis=-1)

    def kl_divergence(self, z, mu, std):
        log_pz = self.normal.log_prob([z, ops.zeros_like(mu), ops.ones_like(std)])
        log_qzx = self.normal.log_prob([z, mu, std])

        # kl
        kl = log_qzx - log_pz
        kl = ops.sum(kl, axis=-1)
        return kl

    def reconstruct(self, x):
        outputs = self.call(x)
        x_mean, x_log_var = outputs["x_mean"], outputs["x_log_var"]
        x_std = ops.exp(0.5 * x_log_var)
        return self.normal([x_mean, x_std])

    def call(self, x):

        # log_enc = []
        # log_dec = []
        kls = [] # (nz, batch_size, seq_len)
        recon_losses = [] # (nz, batch_size, seq_len)

        for i in range(self.nz):
            # ENCODER
            # get the parameters of inference distribution i given x q(z_i|x) or z q(z_i|z_{i+1})
            # (batch_size, seq_len, latent_dim)
            z_mean, z_log_var = self.encoder(x, lvl=i)

            # sample z from q - encoder
            z_std = ops.exp(0.5 * z_log_var)  # convert log_var to std
            z = self.normal([z_mean, z_std])

            # logq - kl
            logq = self.kl_divergence(z, z_mean, z_std)
            kls = logq # cambiar cuando nz > 1

            # DECODER
            # get the parameters of generative distribution i p(x_i|z_i) or z p(z_i|z_{i+1})
            # (batch_size, seq_len, feat_dim)
            x_mean, x_log_var = self.decoder(z, lvl=i)

            # logp - reconstruction loss
            logp = self.gaussian_likelihood(x_mean, x_log_var, x)
            recon_losses = logp # cambiar cuando nz > 1
            print("shape of recon_losses: ", recon_losses.shape)

            # sample from p(x|z) to get x
            #x_recon = self.normal([x_mu, ops.exp(x_log_scale)])

        return {
            "z": z,
            "x_mean": x_mean,
            "x_log_var": x_log_var,
            "kl": kls, 
            "recon_loss": recon_losses
        }

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
        # elbo = recon_loss - (self.beta * kl)
        return ops.mean(elbo)