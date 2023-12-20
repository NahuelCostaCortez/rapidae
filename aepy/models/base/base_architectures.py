"""
Base class for defining the encoder and decoder architectures
"""

import keras as keras
import tensorflow as tf
from keras import layers


class BaseEncoder(layers.Layer):
    """
    Base class for defining encoder architectures. It includes common functionality and attributes
	shared by different encoder implementations.

    Args
    ----
            input_dim (int): The dimensionality of the input data.
            latent_dim (int): The dimensionality of the latent space.
            layers_conf (list): The configuration of layers in the encoder architecture.
            name(str): Name for the instanced encoder.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, name="encoder"):
        """
        Constructor for the encoder.
        """
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")

    def call(self, x):
        """ 
        This function must be implemented in a child class.

        Parameters
        ----------
                x (tf.Tensor): input tensor

        Returns
        -------
                tf.Tensor: the output of the encoder
        """
        raise NotImplementedError


class BaseDecoder(layers.Layer):
    """
    Base class for defining decoder architectures. It includes common functionality and attributes
	shared by different encoder implementations.

    Args
    ----
            input_dim (int): The dimensionality of the input data.
            latent_dim (int): The dimensionality of the latent space.
            layers_conf (list): The configuration of layers in the decoder architecture.
            name(str): Name for the instanced decoder.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, name="decoder"):
        """
        Constructor for the decoder.
        """
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = reversed(layers_conf)

    def call(self, x):
        """ 
        This function must be implemented in a child class 

        Parameters
        ----------
                x (tf.Tensor): input tensor with the latent data

        Returns
        -------
                tf.Tensor: the output of the decoder
        """
        raise NotImplementedError
