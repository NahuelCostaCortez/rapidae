from typing import Tuple, Union

import keras

from rapidae.conf import Logger
from rapidae.models.base import BaseAE


class Beta_VAE(BaseAE):
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
        **kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        beta: float = 1.0,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
        **kwargs,
    ):
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            decoder=decoder,
            layers_conf=layers_conf,
            **kwargs,
        )

        # Initialize sampling layer
        self.sampling = self.Sampling()
        
        if beta < 0.0:
            Logger().log_warning('The beta parameter should not be a value less than or equal to zero.')
        else:
            self.beta = beta

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    # keras model call function
    def call(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling([z_mean, z_log_var])
        outputs = {}
        outputs["z"] = z
        outputs["z_mean"] = z_mean
        outputs["z_log_var"] = z_log_var
        recon_x = self.decoder(z)
        outputs["recon"] = recon_x

        return outputs

    # TODO: change to function named NormalSampler
    class Sampling(keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a sample."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = keras.ops.shape(z_mean)[0]
            dim = keras.ops.shape(z_mean)[1]
            # Added seed for reproducibility
            epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)

            return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # KL loss
        kl_loss = -0.5 * (
            1
            + y_pred["z_log_var"]
            - keras.ops.square(y_pred["z_mean"])
            - keras.ops.exp(y_pred["z_log_var"])
        )
        kl_loss = keras.ops.mean(keras.ops.sum(kl_loss, axis=1))
        self.kl_loss_tracker.update_state(kl_loss)

        # Reconstruction loss
        recon_loss = keras.ops.mean(
            keras.ops.sum(
                keras.losses.binary_crossentropy(x, y_pred["recon"], axis=(1, 2)))  
        )
        self.reconstruction_loss_tracker.update_state(recon_loss)

        loss = (self.beta * kl_loss) + recon_loss

        return loss
