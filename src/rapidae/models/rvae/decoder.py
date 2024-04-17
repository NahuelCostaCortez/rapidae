from rapidae.models.base import BaseDecoder
from keras.layers import RepeatVector, LSTM, Bidirectional


class Decoder(BaseDecoder):
    """
    Decoder from RVAE paper - DOI: 10.1109/ACCESS.2021.3064854

    Args:
        input_dim (tuple): Dimensions of the input data (seq_len, num_features).
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        h_decoded_1 (RepeatVector): RepeatVector layer.
        h_decoded_2 (Bidirectional): Bidirectional LSTM layer.
        h_decoded (LSTM): LSTM layer.
    """

    def __init__(self, input_dim, latent_dim):
        BaseDecoder.__init__(self, input_dim, latent_dim)
        self.seq_len = input_dim[0]
        self.num_features = input_dim[1]
        self.h_decoded_1 = RepeatVector(self.seq_len)
        self.h_decoded_2 = Bidirectional(LSTM(300, return_sequences=True))
        self.h_decoded = LSTM(self.num_features, return_sequences=True)

    def call(self, z):
        """
        Forward pass of the Recurrent Decoder.

        Args:
            z (Tensor): Latent space representation.

        Returns:
            x (Tensor): Decoded output.
        """
        x = self.h_decoded_1(z)
        x = self.h_decoded_2(x)
        x = self.h_decoded(x)

        return x
