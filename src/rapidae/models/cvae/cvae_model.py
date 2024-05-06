from typing import Tuple, Union

from keras.layers import Embedding, Concatenate
from keras import metrics, ops, losses
from rapidae.models.base import BaseAE
from rapidae.models.distributions import Normal


class CVAE(BaseAE):
    """
    Conditional Variational AutoEncoder (CVAE) model

    Args:
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        encoder (BaseEncoder): An instance of BaseEncoder.
        decoder (BaseDecoder): An instance of BaseDecoder.
        n_classes (int): Number of classes.
        embedding_dim (int): Dimension of the embedding space.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        n_classes: int = 6,
        embedding_dim: int = 2,
    ):
        # Initialize base class
        BaseAE.__init__(self, input_dim, latent_dim, encoder, decoder)
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        # Initialize distribution
        self.sampling = Normal()

        # Embedding layer
        self.condition_embedding = Embedding(
            input_dim=n_classes, output_dim=embedding_dim
        )

        # Initialize metrics
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")

    def call(self, inputs):
        x, y = inputs
        # encoder
        encoder_inputs = Concatenate(axis=1)([x, self.condition_embedding(y)])
        z_mean, z_log_var = self.encoder(encoder_inputs)
        z_std = ops.exp(0.5 * z_log_var)
        # sampling
        z = self.sampling([z_mean, z_std])
        # decoder
        decoder_inputs = Concatenate(axis=1)([z, self.condition_embedding(y)])
        x_recon = self.decoder(decoder_inputs)

        return {
            "z": z,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "x_recon": x_recon,
        }

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
        recon_loss = ops.mean(ops.sum(losses.mean_squared_error(x, y_pred["x_recon"])))
        self.reconstruction_loss_tracker.update_state(recon_loss)

        loss = kl_loss + recon_loss

        return loss
