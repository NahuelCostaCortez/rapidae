from rapidae.models.base import BaseDecoder
from keras.layers import Conv2DTranspose


class Decoder(BaseDecoder):
    """
    Decoder architecture for a vector quantization variational autoencoder (VQ-VAE).
    Adapted from Keras tutorial: https://keras.io/examples/generative/vq_vae/

    Args:
        input_dim (tuple): Dimensions of the input data (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of filters in each transposed convolutional layer.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        layers_dict (dict): Dictionary containing the transposed convolutional layers of the decoder.
        layers_idx (list): List of layer indices.
        conv2d_transpose_recons (tf.keras.layers.Conv2DTranspose): Transposed convolutional layer for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[64,32], **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["conv2d_" + str(idx)] = Conv2DTranspose(
                depth, 3, activation="relu", strides=2, padding="same"
            )

        self.conv2d_transpose_recons = Conv2DTranspose(1, 3, padding="same")

    def call(self, x):
        """
        Forward pass of the convolutional VQ decoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder outputs.
        """
        for _, layer in self.layers_dict.items():
            x = layer(x)
        x = self.conv2d_transpose_recons(x)

        return x
