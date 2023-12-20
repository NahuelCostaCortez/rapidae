import tensorflow as tf
import keras
from keras import layers

from aepy.models.base import BaseDecoder, BaseEncoder

# ------------------- VANILLA MLP VAE ------------------- #
class Encoder_MLP(BaseEncoder):
    """
    Vanilla multi-layer perceptron (MLP) encoder architecture for a variational autoencoder (VAE).

    Args
    ----
    input_dim (int): Dimensionality of the input data.
    latent_dim (int): Dimensionality of the latent space.
    layers_conf (list): Configuration of layers in the encoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' +
                             str(idx)] = layers.Dense(depth, activation="relu")

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x_z_mean = self.z_mean(x)
        x_log_var = self.z_log_var(x)
        return x_z_mean, x_log_var


class Decoder_MLP(BaseDecoder):
    """
    Vanilla multi-layer perceptron (MLP) decoder architecture for a variational autoencoder (VAE).

    Args
    ----
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' +
                             str(idx)] = layers.Dense(depth, activation="relu")

        self.dense_recons = layers.Dense(
            self.input_dim[1], activation="sigmoid")

    def call(self, z):
        for name, layer in self.layers_dict.items():
            z = layer(z)
        x = self.dense_recons(z)
        return x
# ------------------------------------------------------------------- #


# --------------- ENCODER-DECODER FROM KERAS TUTORIAL --------------- #
class Encoder_Conv_MNIST(BaseEncoder):
    """
    Convolutional encoder architecture from keras tutorial for a variational autoencoder (VAE).

    Args
    ----
        input_dim (int): Dimensionality of the input data.
        late RecurrentEncodernt_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.
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
    Convolutional decoder architecture from keras tutorial for a variational autoencoder (VAE).

    Args
    ----
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.
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

# ----------------- CONV ENCODER-DECODER FOR VQ-VAE ----------------- #
class Encoder_Conv_VQ_MNIST(BaseEncoder):
    """
    Encoder architecture from keras tutorial for a vector quantization variational autoencoder (VQ-VAE).

    Args
    ----
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['conv2d_' + str(idx)] = layers.Conv2D(depth, 3, activation="relu",
                                                                   strides=2, padding="same")
        # self.flatten = layers.Flatten()
        self.encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.encoder_outputs(x)
        return x


class Decoder_Conv_VQ_MNIST(BaseDecoder):
    """
    Decoder architecture from keras tutorial for a vector quantization variational autoencoder (VQ-VAE).

    Args
    ----
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['conv2d_' + str(idx)] = layers.Conv2DTranspose(depth, 3, activation="relu",
                                                                            strides=2, padding="same")

        self.conv2d_transpose_recons = layers.Conv2DTranspose(
            1, 3, padding="same")

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.conv2d_transpose_recons(x)
        return x

# ------------------------------------------------------------------- #

# ------------------ ENCODER-DECODER FROM RVE paper ----------------- #
class RecurrentEncoder(BaseEncoder):
    """
    Encoder from RVE paper - DOI: 10.1016/j.ress.2022.108353

    Args
    ----
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.
        masking_value (float): Value for masking sequences.
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

    Args
    ----
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.
        masking_value (float): Value for masking sequences.
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


# ---------------------- SPARSE ENCODER-DECODER --------------------- #
class SparseEncoder(BaseEncoder):
    """
    Sparse encoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.lambda_ = 4  # TODO: Make lambda a customizable param
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' + str(idx)] = layers.Dense(depth,
                                                                 activation='sigmoid',
                                                                 kernel_regularizer=keras.regularizers.l2(
                                                                     self.lambda_/2),
                                                                 activity_regularizer=SparseRegularizer())

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        return x


class SparseDecoder(BaseDecoder):
    """
    Sparse decoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.lambda_ = 4  # TODO: Make lambda a customizable param
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' + str(idx)] = layers.Dense(depth,
                                                                 activation='sigmoid',
                                                                 kernel_regularizer=keras.regularizers.l2(
                                                                     self.lambda_/2),
                                                                 activity_regularizer=SparseRegularizer())

        self.dense_recons = layers.Dense(
            self.input_dim[1], activation="sigmoid")

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.dense_recons(x)
        return x


class SparseRegularizer(keras.regularizers.Regularizer):
    """
    Inherits from keras.regularizers.Regularizer and represents a regularization term for sparsity.

    Args
    ----
        p (float): Target sparsity probability. Default is 0.01
        beta (int): Regularization strength. Default is 3.
    """

    def __init__(self, p=0.01, beta=3):
        self.p = p
        self.beta = beta

    def __call__(self, x):
        self.p_hat = tf.keras.backend.mean(x)
        kl_divergence = self.p*(tf.keras.backend.log(self.p/self.p_hat)) + \
            (1-self.p)*(tf.keras.backend.log(1-self.p/1-self.p_hat))
        return self.beta * tf.keras.backend.sum(kl_divergence)

    def get_config(self):
        return {'p': float(self.p),
                'p_hat': float(self.p_hat),
                'beta': float(self.beta)}


"""
def sparse_regularizer(activation_matrix):
    
    #Function to define a custom regularizer based on Kullback-Leibler Divergence.
    
    p = 0.01
    beta = 3
    p_hat = tf.keras.backend.mean(activation_matrix) 
  
    KL_divergence = p*(tf.keras.backend.log(p/p_hat)) + (1-p)*(tf.keras.backend.log(1-p/1-p_hat))
    
    sum = tf.keras.backend.sum(KL_divergence) 
   
    return beta * sum 
"""
# ------------------------------------------------------------------- #

# --------------------- VANILLA ENCODER-DECODER --------------------- #


class VanillaEncoder(layers.Layer):
    """
    Vanilla encoder for normal autoencoder (AE).

    Args
    ----
        input_dim: Dimensionality of the input data.
        latent_dim: Dimensionality of the latent space.
        layers_conf: Configuration of layers in the encoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, name="encoder", **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' +
                             str(idx)] = layers.Dense(depth, activation='relu')

        self.enc_layer = layers.Dense(latent_dim, activation='relu')

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.enc_layer(x)
        return x


class VanillaDecoder(layers.Layer):
    """
    Vanilla decoder for normal autoencoder (AE).

    Args
    ----
        input_dim: Dimensionality of the input data.
        latent_dim: Dimensionality of the latent space.
        layers_conf: Configuration of layers in the decoder architecture.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, name='decoder', **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict['dense_' +
                             str(idx)] = layers.Dense(depth, activation='relu')

        self.dense_recons = layers.Dense(self.input_dim, activation="sigmoid")

    def call(self, x):
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.dense_recons(x)
        return x
# ------------------------------------------------------------------- #


# ------------------------ ADDITIONAL MODULES ----------------------- #
class BaseRegressor(layers.Layer):
    """
    Architecture for simple regressor compound by 2 Dense layers with 200 and 1 cells.
    Currently not customizable.

    Args
    ----
        name (str): Name of the regressor layer.

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

    Args
    ----
        n_classes (int): Number of classes for classification.
        name (str): Name of the classifier layer.
    """

    def __init__(self, n_classes, name='classifier'):
        super().__init__(name=name)
        self.intermediate = layers.Dense(200, activation='relu')
        self.outputs = layers.Dense(
            n_classes, activation='softmax', name='class_output')

    def call(self, x):
        x = self.intermediate(x)
        x = self.outputs(x)
        return x
# ------------------------------------------------------------------- #
