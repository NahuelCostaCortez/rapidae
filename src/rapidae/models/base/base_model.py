"""
Base class for implementing all the models.
"""

from typing import Optional, Tuple, Union

from keras import Model

from rapidae.models.base import BaseEncoder, BaseDecoder
from rapidae.conf import Logger


class BaseAE(Model):
    """
    Base class for for implementing autoencoder-based models.
    It includes functionality for initializing the encoder and decoder,
    as well as defining the forward pass for obtaining the model outputs.

    Args:
        input_dim (Union[Tuple[int, ...], None]): The dimensionality of the input data. Default is None.
        latent_dim (int): The dimensionality of the latent space. Default is 2.
        encoder (BaseEncoder): An instance of BaseEncoder. Default is None.
        decoder (BaseDecoder): An instance of BaseDecoder. Default is None. If model has no encoder this should be set to False.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        kwargs: Optional[dict] = None,
    ):

        super(BaseAE, self).__init__()
        self.logger = Logger()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

        # If no input dimension is provided, raise an error
        if self.input_dim is None:
            raise AttributeError(
                "No input dimension found."
                "\n'input_dim' argument must be provided."
                "\nTypically (width, height, channels) for images or (time_steps, features) for time series)."
            )

        # Try first to set the encoder and decoder from the child class
        if self.encoder is None:
            import importlib

            self.logger.log_info(
                "Trying to set encoder and decoder from child class..."
            )
            parent_class = type(self).__module__.split(".")[-2]
            # Dynamically import the models from the child class
            model_module = importlib.import_module("rapidae.models." + parent_class)
            Encoder = getattr(model_module, "Encoder")
            Decoder = getattr(model_module, "Decoder")

            if Encoder is not None:
                self.encoder = Encoder(self.input_dim, self.latent_dim)
                self.logger.log_info(f"Encoder set from {parent_class}")
            else:
                # This means that the child class does not have an encoder -> use default encoder
                self.logger.log_warning(
                    f"No encoder found in {parent_class}, using default encoder with layers_conf=[512]..."
                )
                from rapidae.models.base import VanillaEncoder

                layers_conf = [512]
                encoder = VanillaEncoder(input_dim, latent_dim, layers_conf=layers_conf)

            if self.decoder != False:
                # This means that the model does have a decoder -> try to set it from the child class
                if Decoder is not None:
                    self.decoder = Decoder(self.input_dim, self.latent_dim)
                    self.logger.log_info(f"Decoder set from {parent_class}")
                else:
                    # This means that the child class does not have a decoder -> use default decoder
                    self.logger.log_warning(
                        f"No decoder found in {parent_class}, using default decoder with layers_conf=[512]..."
                    )
                    from rapidae.models.base import VanillaDecoder

                    layers_conf = [512]
                    decoder = VanillaDecoder(
                        input_dim, latent_dim, layers_conf=layers_conf
                    )
        else:
            self.logger.log_info("Using provided encoder")
            if self.decoder is not False and self.decoder is not None:
                self.logger.log_info("Using provided decoder")

    def call(self, inputs):
        """
        Forward pass.
        This function must be implemented in a child class.

        Args:
            inputs (Tensor): Input data.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError
