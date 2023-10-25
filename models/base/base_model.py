"""
Base class for implementing all the models.
"""

from typing import Optional, Tuple, Union

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
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder:  Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        uses_default_encoder: bool = True,
        uses_default_decoder: bool = True,
        layers_conf: list = None,
        **kwargs
    ):
    
        super(BaseAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if layers_conf is None:
            Logger().log_warning('No specific layer configuration has been provided. Creating default configuration...')
            self.layers_conf = [512]
        else:
            self.layers_conf = layers_conf
        Logger().log_info('The selected configuration of layers for the autoencoder is {}'.format(self.layers_conf))

        if encoder is None:
            Logger().log_warning('No encoder provided, using default MLP encoder')
            if self.input_dim is None:
                raise AttributeError(
                    "No input dimension provided!"
                    "'input_dim' must be provided in the model config"
                )
            self.encoder = Encoder_MLP(self.input_dim, self.latent_dim, self.layers_conf)
        self.encoder = encoder(self.input_dim, self.latent_dim, self.layers_conf, **kwargs)

        if decoder is None:
            Logger().log_warning('No decoder provider, using default MLP decoder'),
            if self.input_dim is None:
                raise AttributeError(
                    "No input dimension provided!"
                    "'input_dim' must be provided in the model config"
                )
            self.decoder = Decoder_MLP(self.input_dim, self.latent_dim, self.layers_conf)
        self.decoder = decoder(self.input_dim, self.latent_dim, self.layers_conf, **kwargs)

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
