from rapidae.models.base import BaseEncoder
from keras.layers import Masking, LSTM, Bidirectional, Dense


class Encoder(BaseEncoder):
    """
    Encoder from RVAE paper - DOI: 10.1109/ACCESS.2021.3064854

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        h (keras.layers.Bidirectional): Bidirectional LSTM layer.
        z_mean (keras.layers.Dense): Dense layer for the mean.
        z_log_var (keras.layers.Dense): Dense layer for the log variance.
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.masking_value = (
            kwargs["masking_value"] if "masking_value" in kwargs else -99.0
        )
        self.mask = Masking(mask_value=self.masking_value)
        self.h = Bidirectional(LSTM(300))
        self.z_mean = Dense(self.latent_dim, name="z_mean")
        self.z_log_var = Dense(self.latent_dim, name="z_log_var")

    def call(self, x):
        """
        Forward pass of the Recurrent Encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x_z_mean (Tensor), x_z_log_var (Tensor): Tuple containing the mean and log variance.
        """
        x = self.mask(x)
        x = self.h(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        return z_mean, z_log_var
