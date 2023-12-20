"""
Base class for implementing all the models.
"""

from typing import Optional, Tuple, Union

import keras
import tensorflow as tf

from rapidae.models.base.default_architectures import BaseDecoder, BaseEncoder, Decoder_MLP, Encoder_MLP
from rapidae.conf import Logger


class BaseAE(keras.Model):
    """
    Base class for for implementing various autoencoder-based models. It includes functionality for
    initializing the encoder and decoder, as well as defining the main call pass for obtaining the model outputs during
    inference. Child classes should implement the 'call', 'train_step', and 'test_step' methods for specific use cases.

    Args
    ----
        input_dim (Union[Tuple[int, ...], None]): The dimensionality of the input data. Default is None.
        latent_dim (int): The dimensionality of the latent space. Default is 2.
        encoder (BaseEncoder): An instance of BaseEncoder. Default is None
        decoder (BaseDecoder): An instance of BaseDecoder. Default is None
        uses_default_encoder (bool): Flag indicating whether the default MLP encoder should be used. Default is True.
        uses_default_decoder (bool): Flag indicating whether the default MLP decoder should be used. Default is True.
        layers_conf (list): The configuration of layers in the encoder and decoder architectures. Default is None.
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
            Logger().log_warning(
                'No specific layer configuration has been provided. Creating default configuration...')
            self.layers_conf = [512]
        else:
            self.layers_conf = layers_conf

        if encoder is None:
            Logger().log_warning('No encoder provided, using default MLP encoder')
            if self.input_dim is None:
                raise AttributeError(
                    "No input dimension provided!"
                    "'input_dim' must be provided in the model config"
                )
            self.encoder = Encoder_MLP(
                self.input_dim, self.latent_dim, self.layers_conf)
        self.encoder = encoder(
            self.input_dim, self.latent_dim, self.layers_conf, **kwargs)

        if decoder is None:
            Logger().log_warning('No decoder provider, using default MLP decoder'),
            if self.input_dim is None:
                raise AttributeError(
                    "No input dimension provided!"
                    "'input_dim' must be provided in the model config"
                )
            self.decoder = Decoder_MLP(
                self.input_dim, self.latent_dim, self.layers_conf)
        self.decoder = decoder(
            self.input_dim, self.latent_dim, self.layers_conf, **kwargs)

    # make the class callable
    # def __call__(self, *args, **kwargs):
    #    return self

    #@tf.function
    def call(self, inputs):
        """
        Main call pass outputing the VAE outputs.
        This function must be implemented in a child class.
        """
        raise NotImplementedError

    #@tf.function
    #def train_step(self, inputs):
    #    #raise NotImplementedError
    #    pass

    #                                            
    #def test_step(self, inputs):
    #    #raise NotImplementedError
    #    pass
