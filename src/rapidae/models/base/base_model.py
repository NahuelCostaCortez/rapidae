"""
Base class for implementing all the models.
"""

from typing import Optional, Tuple, Union

import keras

from rapidae.models.base.default_architectures import (
    BaseDecoder,
    BaseEncoder,
    Decoder_MLP,
    Encoder_MLP,
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
        exclude_decoder: bool = False,
        layers_conf: list = None,
        **kwargs,
    ):
        super(BaseAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.exclude_decoder = exclude_decoder

        if layers_conf is None:
            Logger().log_warning(
                "No specific layer configuration has been provided. Creating default configuration..."
            )
            self.layers_conf = [512]
        else:
            self.layers_conf = layers_conf

        if encoder is None:
            Logger().log_warning("No encoder provided, using default MLP encoder")
            if self.input_dim is None:
                raise AttributeError(
                    "No input dimension provided!"
                    "'input_dim' must be provided in the model config"
                )
            self.encoder = Encoder_MLP(
                self.input_dim, self.latent_dim, self.layers_conf
            )
        self.encoder = encoder(
            self.input_dim, self.latent_dim, self.layers_conf, **kwargs
        )

        if not self.exclude_decoder:
            if decoder is None:
                Logger().log_warning("No decoder provider, using default MLP decoder"),
                if self.input_dim is None:
                    raise AttributeError(
                        "No input dimension provided!"
                        "'input_dim' must be provided in the model config"
                    )
                self.decoder = Decoder_MLP(
                    self.input_dim, self.latent_dim, self.layers_conf
                )
            else:
                self.decoder = decoder(
                    self.input_dim, self.latent_dim, self.layers_conf, **kwargs
                )

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
