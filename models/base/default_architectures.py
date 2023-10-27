import tensorflow as tf
from tensorflow.keras import layers

from models.base import BaseDecoder, BaseEncoder

# ------------------- VANILLA MLP ENCODER-DECODER ------------------- #
class Encoder_MLP(BaseEncoder):
    """
    Vanilla MLP encoder
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' + str(idx)] = layers.Dense(depth, activation="relu")
            
    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x_z_mean = self.z_mean(x)
        x_log_var = self.z_log_var(x)
        return x_z_mean, x_log_var


class Decoder_MLP(BaseDecoder):
    """
    Vanilla MLP decoder
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' + str(idx)] = layers.Dense(depth, activation="relu")

        self.dense_recons = layers.Dense(self.input_dim[1], activation="sigmoid")

    def call(self, z):
        for name, layer in self.layers_dict.items():
            z = layer(z)
        x = self.dense_recons(z)
        return x
# ------------------------------------------------------------------- #


# --------------- ENCODER-DECODER FROM KERAS TUTORIAL --------------- #
class Encoder_Conv_MNIST(BaseEncoder):
    """
    Encoder architecture from keras tutorial
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['conv2d_' + str(idx)] = layers.Conv2D(depth, 3, activation="relu",
                                                                 strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16, activation="relu")
        
    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x_z_mean = self.z_mean(x)
        x_log_var = self.z_log_var(x)
        return x_z_mean, x_log_var


class Decoder_Conv_MNIST(BaseDecoder):
    """
    Decoder architecture from keras tutorial
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.dense = layers.Dense(7 * 7 * 64, activation="relu")
        self.reshape = layers.Reshape((7, 7, 64))
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['conv2d_' + str(idx)] = layers.Conv2DTranspose(depth, 3, activation="relu",
                                                                 strides=2, padding="same")

        self.conv2d_transpose_recons = layers.Conv2DTranspose(
            1, 3, activation="sigmoid", padding="same")

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.conv2d_transpose_recons(x)
        return x
# ------------------------------------------------------------------- #


# ------------------ ENCODER-DECODER FROM RVE paper ----------------- #
class RecurrentEncoder(BaseEncoder):
    """
    Encoder from RVE paper - DOI: 10.1016/j.ress.2022.108353
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
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

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
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

    def __init__(self, input_dim, latent_dim, layers_conf, name="encoder", **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' + str(idx)] = layers.Dense(depth, activation='relu')
    
    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        return x

class VanillaDecoder(layers.Layer):
    """
    Vanilla decoder for normal AE
    """

    def __init__(self, input_dim, latent_dim, layers_conf, name='decoder', **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' + str(idx)] = layers.Dense(depth, activation='relu')

        self.dense_recons = layers.Dense(self.input_dim[1], activation="sigmoid")

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.dense_recons(x)
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
    
class BaseClassifier(layers.Layer):
    """
    Simple classifier to be applied to the latent space of an AE
    """

    def __init__(self, n_classes, name='classifier'):
        super().__init__(name=name)
        self.intermediate = layers.Dense(200, activation='relu')
        self.outputs = layers.Dense(n_classes, activation='softmax', name='class_output')

    def call(self, x):
        x = self.intermediate(x)
        x = self.outputs(x)
        return x
# ------------------------------------------------------------------- #
