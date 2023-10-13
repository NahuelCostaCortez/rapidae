"""
Base class for implementing all the models.
"""

from typing import Optional

import tensorflow as tf

from .default_architectures import BaseDecoder, BaseEncoder, Decoder_MLP, Encoder_MLP
from conf import Logger


class BaseAE(tf.keras.Model):
    """
    Base class for Autoencoder based models.

    Args
    ----
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """

    def __init__(
        self,
        model_config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        super(BaseAE, self).__init__()
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim

        if encoder is None:
            Logger.log_warning('No encoder provided, using default MLP encoder')
            if self.input_dim is None:
                raise AttributeError(
                    "No inpu)t dimension provided!"
                    "'input_dim' must be provided in the model config"
                )
            self.encoder = Encoder_MLP(model_config)
        self.encoder = encoder

        if decoder is None:
            Logger.log_warning('No decoder provider, using default MLP decoder')
            if self.input_dim is None:
                raise AttributeError(
                    "No input dimension provided!"
                    "'input_dim' must be provided in the model config"
                )
            self.decoder = Decoder_MLP(model_config)
        self.decoder = decoder

    # make the class callable
    # def __call__(self, *args, **kwargs):
    #    return self

    @tf.function
    def call(self, inputs):
        """
        Main call pass outputing the VAE outputs.
        This function must be implemented in a child class.
        """
        raise NotImplementedError

    @tf.function
    def train_step(self, inputs):
        raise NotImplementedError

    @tf.function
    def test_step(self, inputs):
        raise NotImplementedError
