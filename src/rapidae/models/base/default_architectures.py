import keras
from keras import layers

from rapidae.models.base import BaseDecoder, BaseEncoder


# ------------------- VANILLA MLP VAE ------------------- #
class VAE_Encoder_MLP(BaseEncoder):
    """
    Vanilla multi-layer perceptron (MLP) encoder architecture for a variational autoencoder (VAE).

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of neurons in each layer.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        layers_dict (dict): Dictionary containing the layers of the MLP.
        layers_idx (list): List of layer indices.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="relu"
            )

        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")

    def call(self, x):
        """
        Forward pass of the MLP encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x_z_mean (Tensor), x_log_var (Tensor): Tuple containing the mean and log variance of the latent space.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x_z_mean = self.z_mean(x)
        x_log_var = self.z_log_var(x)

        return x_z_mean, x_log_var


class VAE_Decoder_MLP(BaseDecoder):
    """
    Multi-Layer Perceptron Decoder.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of neurons in each layer.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        layers_dict (dict): Dictionary containing the layers of the MLP.
        layers_idx (list): List of layer indices.
        dense_recons (keras.layers.Dense): Dense layer for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="relu"
            )

        self.dense_recons = layers.Dense(self.input_dim, activation="sigmoid")

    def call(self, z):
        """
        Forward pass of the MLP decoder.

        Args:
            z (Tensor): Latent space representation.

        Returns:
            x (Tensor): Reconstructed output.
        """
        for name, layer in self.layers_dict.items():
            z = layer(z)
        x = self.dense_recons(z)

        return x


# ------------------------------------------------------------------- #


# --------------- ENCODER-DECODER FROM KERAS TUTORIAL --------------- #
class VAE_Encoder_Conv_MNIST(BaseEncoder):
    """
    Convolutional encoder architecture from keras tutorial for a variational autoencoder (VAE).

    Args:
        input_dim (tuple): Dimensions of the input data (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of filters in each convolutional layer.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        layers_dict (dict): Dictionary containing the convolutional layers of the encoder.
        layers_idx (list): List of layer indices.
        flatten (keras.layers.Flatten): Flatten layer for converting convolutional output to a vector.
        dense (keras.layers.Dense): Dense layer in the encoder.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["conv2d_" + str(idx)] = layers.Conv2D(
                depth, 3, activation="relu", strides=2, padding="same"
            )
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16, activation="relu")
        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")

    def call(self, x):
        """
        Forward pass of the convolutional encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x_z_mean (Tensor), x_log_var (Tensor): Tuple containing the mean and log variance of the latent space.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x_z_mean = self.z_mean(x)
        x_log_var = self.z_log_var(x)

        return x_z_mean, x_log_var


class VAE_Decoder_Conv_MNIST(BaseDecoder):
    """
    Convolutional encoder architecture from keras tutorial for a variational autoencoder (VAE).

    Args:
        input_dim (tuple): Dimensions of the input data (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of filters in each transposed convolutional layer.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        dense (keras.layers.Dense): Dense layer in the decoder.
        reshape (keras.layers.Reshape): Reshape layer to convert flattened input to 4D tensor.
        layers_dict (dict): Dictionary containing the transposed convolutional layers of the decoder.
        layers_idx (list): List of layer indices.
        conv2d_transpose_recons (keras.layers.Conv2DTranspose): Transposed convolutional layer for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.dense = layers.Dense(7 * 7 * 64, activation="relu")
        self.reshape = layers.Reshape((7, 7, 64))
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["conv2d_" + str(idx)] = layers.Conv2DTranspose(
                depth, 3, activation="relu", strides=2, padding="same"
            )

        self.conv2d_transpose_recons = layers.Conv2DTranspose(
            1, 3, activation="sigmoid", padding="same"
        )

    def call(self, z):
        """
        Forward pass of the convolutional decoder.

        Args:
            z (Tensor): Latent space representation.

        Returns:
            x (Tensor): Reconstructed output.
        """
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

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["conv2d_" + str(idx)] = layers.Conv2D(
                depth, 3, activation="relu", strides=2, padding="same"
            )
        self.encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")

    def call(self, x):
        """
        Forward pass of the convolutional VQ encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder outputs.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.encoder_outputs(x)

        return x


class Decoder_Conv_VQ_MNIST(BaseDecoder):
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

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["conv2d_" + str(idx)] = layers.Conv2DTranspose(
                depth, 3, activation="relu", strides=2, padding="same"
            )

        self.conv2d_transpose_recons = layers.Conv2DTranspose(1, 3, padding="same")

    def call(self, x):
        """
        Forward pass of the convolutional VQ decoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder outputs.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.conv2d_transpose_recons(x)

        return x


# ------------------------------------------------------------------- #


# ------------------ ENCODER-DECODER FROM RVE paper ----------------- #
class RecurrentEncoder(BaseEncoder):
    """
    Encoder from RVE paper - DOI: 10.1016/j.ress.2022.108353

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        masking_value (float): Value for masking sequences.
        mask (keras.layers.Masking): Masking layer.
        h (keras.layers.Bidirectional): Bidirectional LSTM layer.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.masking_value = (
            kwargs["masking_value"] if "masking_value" in kwargs else -99.0
        )
        self.mask = layers.Masking(mask_value=self.masking_value)
        self.h = layers.Bidirectional(layers.LSTM(300))
        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")

    def call(self, x):
        """
        Forward pass of the Recurrent Encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x_z_mean (Tensor), x_z_log_var (Tensor): Tuple containing the mean and log variance of the latent space.
        """
        x = self.mask(x)
        x = self.h(x)
        x_z_mean = self.z_mean(x)
        x_z_log_var = self.z_log_var(x)

        return x_z_mean, x_z_log_var


class RecurrentDecoder(BaseDecoder):
    """
    Decoder from RVE paper - DOI: 10.1016/j.ress.2022.108353

    Args:
        input_dim (tuple): Dimensions of the input data (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.

    Attributes:
        h_decoded_1 (keras.layers.RepeatVector): RepeatVector layer in the decoder.
        h_decoded_2 (keras.layers.Bidirectional): Bidirectional LSTM layer in the decoder.
        h_decoded (keras.layers.LSTM): LSTM layer in the decoder.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.h_decoded_1 = layers.RepeatVector(input_dim[0])
        self.h_decoded_2 = layers.Bidirectional(layers.LSTM(300, return_sequences=True))
        self.h_decoded = layers.LSTM(input_dim[1], return_sequences=True)

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


# ------------------------------------------------------------------- #


# ---------------------- SPARSE ENCODER-DECODER --------------------- #
class SparseEncoder(BaseEncoder):
    """
    Sparse Encoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.
        kwargs (dict): Additional keyword arguments.

    Attributes:
        beta (float): Regularization parameter for the sparsity constraint.
        p (float): Target sparsity level for the encoder outputs.
        layers_dict (dict): Dictionary containing the dense layers of the encoder.
        layers_idx (list): List of layer indices.
        encoder_outputs (tf.keras.layers.Dense): Dense layer in the encoder with sparsity regularization.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.beta = kwargs["beta"] if "beta" in kwargs else 3
        self.p = kwargs["p"] if "p" in kwargs else 0.01
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="sigmoid"
            )
        self.encoder_outputs = layers.Dense(
            latent_dim,
            activation="sigmoid",
            activity_regularizer=SparseRegularizer(self.p, self.beta),
        )

    def call(self, x):
        """
        Forward pass of the Sparse Encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder outputs.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.encoder_outputs(x)

        return x


class SparseDecoder(BaseDecoder):
    """
    Sparse Decoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.

    Attributes:
        layers_dict (dict): Dictionary containing the dense layers of the decoder.
        layers_idx (list): List of layer indices.
        dense_recons (keras.layers.Dense): Dense layer in the decoder for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="sigmoid"
            )

        self.dense_recons = layers.Dense(self.input_dim, activation="sigmoid")

    def call(self, x):
        """
        Forward pass of the Sparse Decoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Reconstructed output.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.dense_recons(x)

        return x


class SparseRegularizer(keras.regularizers.Regularizer):
    """
    Sparse Regularizer for enforcing sparsity in a layer.

    Args:
        p (float): Target sparsity level.
        beta (float): Regularization strength.

    Attributes:
        p (float): Target sparsity level.
        beta (float): Regularization strength.
        p_hat (Tensor): Estimated sparsity level.
    """

    def __init__(self, p=0.01, beta=2):
        self.p = p
        self.beta = beta

    def __call__(self, x):
        """
        Computes the regularization term.

        Args:
            x (Tensor): Input tensor.

        Returns:
            x (Tensor): Regularization term.
        """
        self.p_hat = keras.ops.mean(x)
        kl_divergence = self.p * (keras.ops.log(self.p / self.p_hat)) + (1 - self.p) * (
            keras.ops.log(1 - self.p / 1 - self.p_hat)
        )

        return self.beta * keras.ops.sum(kl_divergence)

    def get_config(self):
        """
        Returns the configuration of the regularizer.

        Returns:
            (dict): Configuration dictionary.
        """
        return {
            "p": float(self.p),
            "p_hat": float(self.p_hat),
            "beta": float(self.beta),
        }


# ------------------------------------------------------------------- #


# --------------------- VANILLA ENCODER-DECODER --------------------- #


class VanillaEncoder(layers.Layer):
    """
    Vanilla Encoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.
        name (str): Name of the encoder layer.

    Attributes:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.
        layers_dict (dict): Dictionary containing the dense layers of the encoder.
        layers_idx (list): List of layer indices.
        enc_layer (keras.layers.Dense): Dense layer in the encoder.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, name="encoder", **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="relu"
            )

        self.enc_layer = layers.Dense(latent_dim, activation="relu")

    def call(self, x):
        """
        Forward pass of the Vanilla Encoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder outputs.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.enc_layer(x)

        return x


class VanillaDecoder(layers.Layer):
    """
    Vanilla Decoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.
        name (str): Name of the decoder layer.

    Attributes:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.
        layers_dict (dict): Dictionary containing the dense layers of the decoder.
        layers_idx (list): List of layer indices.
        dense_recons (keras.layers.Dense): Dense layer in the decoder for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf, name="decoder", **kwargs):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers_conf = layers_conf
        self.layers_dict = {}

        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="relu"
            )

        self.dense_recons = layers.Dense(self.input_dim, activation="sigmoid")

    def call(self, x):
        """
        Forward pass of the Vanilla Decoder.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Reconstructed output.
        """
        for name, layer in self.layers_dict.items():
            x = layer(x)
        x = self.dense_recons(x)

        return x


# ------------------------------------------------------------------- #


# ------------------------ ADDITIONAL MODULES ----------------------- #
class BaseRegressor(layers.Layer):
    """
    Base Regressor class.
    Currently not customizable via layers_conf.

    Args:
        name (str): Name of the regressor layer.

    Attributes:
        dense (keras.layers.Dense): Dense layer in the regressor.
        out (keras.layers.Dense): Output layer in the regressor.
    """

    def __init__(self, name="regressor"):
        super().__init__(name=name)
        self.dense = layers.Dense(200, activation="tanh")
        self.out = layers.Dense(1, name="reg_output")

    def call(self, x):
        """
        Forward pass of the Base Regressor.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Regression output.
        """
        x = self.dense(x)
        x = self.out(x)

        return x


class BaseClassifier(layers.Layer):
    """
    Simple classifier to be applied to the latent space of an Autoencoder.

    Args:
        n_classes (int): Number of classes in the classification task.
        name (str): Name of the classifier layer.

    Attributes:
        intermediate (tf.keras.layers.Dense): Intermediate dense layer in the classifier.
        outputs (tf.keras.layers.Dense): Output layer in the classifier.
    """

    def __init__(self, n_classes, name="classifier"):
        super().__init__(name=name)
        self.intermediate = layers.Dense(200, activation="relu")
        self.outputs = layers.Dense(
            n_classes, activation="softmax", name="class_output"
        )

    def call(self, x):
        """
        Forward pass of the Base Classifier.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Classification output probabilities.
        """
        x = self.intermediate(x)
        x = self.outputs(x)

        return x


# ------------------------------------------------------------------- #
