"""
Base class for implementing all the models.
"""

from typing import Optional, Tuple, Union

import keras

from rapidae.models.base.default_architectures import (
    BaseEncoder,
    BaseDecoder,
    VanillaEncoder,
    VanillaDecoder,
)
from rapidae.conf import Logger


class BaseAE(keras.Model):
    """
    Base class for for implementing various autoencoder-based models.
    It includes functionality for initializing the encoder and decoder,
    as well as defining the main call pass for obtaining the model outputs
    during inference.

    Args:
        input_dim (Union[Tuple[int, ...], None]): The dimensionality of the input data. Default is None.
        latent_dim (int): The dimensionality of the latent space. Default is 2.
        encoder (BaseEncoder): An instance of BaseEncoder. Default is None.
        decoder (BaseDecoder): An instance of BaseDecoder. Default is None.
        exclude_decoder (bool): Whether to exclude the decoder from the model. Default is False.
        layers_conf (list): The configuration of layers in the encoder and decoder architectures. Default is None.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        exclude_decoder: bool = None,
        layers_conf: list = None,
        **kwargs,
    ):
        super(BaseAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.exclude_decoder = exclude_decoder

        # If no input dimension is provided, raise an error
        if self.input_dim is None:
            raise AttributeError(
                "No input dimension found."
                "\n'input_dim' argument must be provided."
                "\nTypically (width, height, channels) for images or (time_steps, features) for time series)."
            )

        # If no encoder or decoder is provided, use default MLP architectures
        if encoder is None and decoder is None:
            if layers_conf is None:
                Logger().log_warning(
                    "No specific layer configuration has been provided. Creating default configuration: [512]..."
                )
                layers_conf = [512]

        if encoder is None:
            Logger().log_warning("No encoder provided, using default encoder")
            encoder = VanillaEncoder

        if decoder is None and self.exclude_decoder == False:
            Logger().log_warning("No decoder provided, using default decoder")
            decoder = VanillaDecoder

        # check if the encoder requires a layers_conf argument
        if "layers_conf" in encoder.__init__.__code__.co_varnames:
            if layers_conf is None:
                Logger().log_warning(
                    "No specific layer configuration has been provided. Creating default configuration: [512]..."
                )
                layers_conf = [512]
            self.encoder = encoder(
                self.input_dim, self.latent_dim, layers_conf, **kwargs
            )
        else:
            self.encoder = encoder(self.input_dim, self.latent_dim, **kwargs)

        if decoder != None:
            # check if the decoder requires a layers_conf argument
            if "layers_conf" in decoder.__init__.__code__.co_varnames:
                self.decoder = decoder(
                    self.input_dim, self.latent_dim, layers_conf, **kwargs
                )
            else:
                self.decoder = decoder(self.input_dim, self.latent_dim, **kwargs)

    def call(self, inputs):
        """
        Forward pass of the base decoder.
        This function must be implemented in a child class.

        Args:
            inputs (Tensor): Input data.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError
