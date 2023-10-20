import tensorflow as tf
from tensorflow.keras import layers

from models.base import BaseDecoder, BaseEncoder

# TODO: Hacer dinámica la creación de capas en decoders y encoders
# ------------------- VANILLA MLP ENCODER-DECODER ------------------- #
class Encoder_MLP(BaseEncoder):
    """
    Vanilla MLP encoder
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.dense = layers.Dense(512, activation="relu")

    def call(self, x):
        x = self.dense(x)
        x_z_mean = self.z_mean(x)
        x_log_var = self.z_log_var(x)
        return x_z_mean, x_log_var


class Decoder_MLP(BaseDecoder):
    """
    Vanilla MLP decoder
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim)
        self.dense = layers.Dense(512, activation="relu")
        self.dense2 = layers.Dense(self.input_dim[1], activation="sigmoid")

    def call(self, z):
        x = self.dense(z)
        x = self.dense2(x)
        return x
# ------------------------------------------------------------------- #


# --------------- ENCODER-DECODER FROM KERAS TUTORIAL --------------- #
class Encoder_Conv_MNIST(BaseEncoder):
    """
    Encoder architecture from keras tutorial
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.conv2d_1 = layers.Conv2D(
            32, 3, activation="relu", strides=2, padding="same")
        self.conv2d_2 = layers.Conv2D(
            64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16, activation="relu")

    def call(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x_z_mean = self.z_mean(x)
        x_log_var = self.z_log_var(x)
        return x_z_mean, x_log_var


class Decoder_Conv_MNIST(BaseDecoder):
    """
    Decoder architecture from keras tutorial
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim)
        self.dense = layers.Dense(7 * 7 * 64, activation="relu")
        self.reshape = layers.Reshape((7, 7, 64))
        self.conv2d_transpose_1 = layers.Conv2DTranspose(
            64, 3, activation="relu", strides=2, padding="same")
        self.conv2d_transpose_2 = layers.Conv2DTranspose(
            32, 3, activation="relu", strides=2, padding="same")
        self.conv2d_transpose_3 = layers.Conv2DTranspose(
            1, 3, activation="sigmoid", padding="same")

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.conv2d_transpose_1(x)
        x = self.conv2d_transpose_2(x)
        x = self.conv2d_transpose_3(x)
        return x
# ------------------------------------------------------------------- #


# ------------------ ENCODER-DECODER FROM RVE paper ----------------- #
class RecurrentEncoder(BaseEncoder):
    """
    Encoder from RVE paper - DOI: 10.1016/j.ress.2022.108353
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.mask = layers.Masking(mask_value=kwargs['masking_value'])
        self.h = layers.Bidirectional(layers.LSTM(300))

    def call(self, x):
        x = self.mask(x)
        x = self.h(x)
        x_z_mean = self.z_mean(x)
        x_z_log_var = self.z_log_var(x)
        return x_z_mean, x_z_log_var


class RecurrentDecoder(BaseDecoder):
    """
    Decoder from RVE paper - DOI: 10.1016/j.ress.2022.108353
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim)
        self.h_decoded_1 = layers.RepeatVector(input_dim[0])
        self.h_decoded_2 = layers.Bidirectional(
            layers.LSTM(300, return_sequences=True))
        self.h_decoded = layers.LSTM(input_dim[1], return_sequences=True)

    def call(self, z):
        x = self.h_decoded_1(z)
        x = self.h_decoded_2(x)
        x = self.h_decoded(x)
        return x
# ------------------------------------------------------------------- #


# ---------------------  VANILLA ENCODER-DECODER -------------------- #
class VanillaEncoder(layers.Layer):
    """
    Vanilla encoder for normal AE
    """

    def __init__(self, input_dim, latent_dim, name="encoder", **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dense = layers.Dense(512, activation="relu")
    
    def call(self, x):
        x = self.dense(x)
        return x

class VanillaDecoder(layers.Layer):
    """
    Vanilla decoder for normal AE
    """

    def __init__(self, input_dim, latent_dim, name='decoder', **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dense = layers.Dense(512, activation="relu")
        self.dense2 = layers.Dense(self.input_dim[1], activation="sigmoid")

    def call(self, x):
        x = self.dense(x)
        x = self.dense2(x)
        return x
# ------------------------------------------------------------------- #


# ------------------------ ADDITIONAL MODULES ----------------------- #
class BaseRegressor(layers.Layer):
    """
    Simple regressor
    """

    def __init__(self, name="regressor"):
        super().__init__(name=name)
        self.dense = layers.Dense(200, activation='tanh')
        self.out = layers.Dense(1, name='reg_output')

    def call(self, x):
        x = self.dense(x)
        x = self.out(x)
        return x
# ------------------------------------------------------------------- #
