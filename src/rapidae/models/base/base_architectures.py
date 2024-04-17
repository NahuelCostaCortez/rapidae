"""
Base class for defining the encoder and decoder architectures
"""

from keras import layers


class BaseEncoder(layers.Layer):
    """
    Base class for defining encoder architectures. It includes common functionality and attributes
        shared by different encoder implementations.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of neurons in each layer.
        name (str): Name of the encoder layer.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=None, name="encoder"):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf

    def call(self, x, **kwargs):
        """
        Forward pass of the base encoder.

        Args:
            x (Tensor): Input data.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError


class BaseDecoder(layers.Layer):
    """
    Base class for defining decoder architectures. It includes common functionality and attributes
        shared by different decoder implementations.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of neurons in each layer.
        name (str): Name of the decoder layer.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=None, name="decoder"):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # The decoder usually uses a reversed architecture of the encoder
        if layers_conf is not None:
            self.layers_conf = layers_conf[::-1]

    def call(self, x):
        """
        Forward pass of the base decoder.

        Args:
            x (Tensor): Input data.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError
