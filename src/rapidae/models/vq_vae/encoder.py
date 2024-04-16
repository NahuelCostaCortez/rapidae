from rapidae.models.base import BaseEncoder
from keras.layers import Conv2D


class Encoder(BaseEncoder):
    """
    Encoder architecture for a vector quantization variational autoencoder (VQ-VAE).
    Adapted from Keras tutorial: https://keras.io/examples/generative/vq_vae/

    Args:
        input_dim (tuple): Dimensions of the input data (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of filters in each convolutional layer.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        layers_dict (dict): Dictionary containing the convolutional layers of the encoder.
        layers_idx (list): List of layer indices.
        encoder_outputs (tf.keras.layers.Conv2D): Convolutional layer for encoder outputs.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[32,64], **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["conv2d_" + str(idx)] = Conv2D(
                depth, 3, activation="relu", strides=2, padding="same"
            )
        self.encoder_outputs = Conv2D(latent_dim, 1, padding="same")

    def call(self, x):
        """
        Forward pass of the convolutional VQ encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder outputs.
        """
        for _, layer in self.layers_dict.items():
            x = layer(x)
        x = self.encoder_outputs(x)

        return x
