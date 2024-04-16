from keras import layers, ops
from keras.regularizers import Regularizer

from rapidae.models.base import BaseDecoder, BaseEncoder

# --------------------- VANILLA ENCODER-DECODER --------------------- #


class VanillaEncoder(BaseEncoder):
    """
    Vanilla Encoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the encoder architecture.

    Attributes:
        layers_dict (dict): Dictionary containing the dense layers.
        layers_idx (list): List of layer indices.
        enc_layer (Dense): Dense layer.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[512]):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)

        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="relu"
            )

        self.enc_layer = layers.Dense(latent_dim, activation="relu")

    def call(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder outputs.
        """
        for _, layer in self.layers_dict.items():
            x = layer(x)
        z = self.enc_layer(x)

        return z


class VanillaDecoder(BaseDecoder):
    """
    Vanilla Decoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Configuration of layers in the decoder architecture.

    Attributes:
        layers_dict (dict): Dictionary containing the dense layers of the decoder.
        layers_idx (list): List of layer indices.
        dense_recons (Dense): Dense layer in the decoder for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[512]):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)

        self.layers_dict = {}
        self.layers_idx = [i for i in range(len(layers_conf))]

        for depth, idx in zip(self.layers_conf, self.layers_idx):
            self.layers_dict["dense_" + str(idx)] = layers.Dense(
                depth, activation="relu"
            )

        self.dense_recons = layers.Dense(self.input_dim, activation="sigmoid")

    def call(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Reconstructed input.
        """
        for _, layer in self.layers_dict.items():
            x = layer(x)
        x = self.dense_recons(x)

        return x


# ------------------------------------------------------------------- #


# ------------------- VANILLA MLP ENCODER-DECODER ------------------- #
class VAE_Encoder_MLP(BaseEncoder):
    """
    Vanilla multi-layer perceptron (MLP) encoder architecture for a variational autoencoder (VAE).

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): List of integers specifying the number of neurons in each layer.

    Attributes:
        layers_dict (dict): Dictionary containing the layers of the MLP.
        layers_idx (list): List of layer indices.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[64]):
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
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            z_mean (Tensor), log_var (Tensor): Tuple containing the mean and log variance.
        """
        for _, layer in self.layers_dict.items():
            x = layer(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        return z_mean, z_log_var


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
        dense_recons (Dense): Dense layer for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[64]):
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
        Forward pass.

        Args:
            z (Tensor): Latent space representation.

        Returns:
            x (Tensor): Reconstructed input.
        """
        for _, layer in self.layers_dict.items():
            z = layer(z)
        x = self.dense_recons(z)

        return x


# ------------------------------------------------------------------- #


# --------------- ENCODER-DECODER FROM KERAS TUTORIAL --------------- #
class VAE_Encoder_Conv_MNIST(BaseEncoder):
    """
    Convolutional encoder architecture from the VAE keras tutorial: https://keras.io/examples/generative/vae/

    Args:
        input_dim (tuple): Dimensions of the input data (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, input_dim, latent_dim):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.conv1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16, activation="relu")
        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")

    def call(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            z_mean (Tensor), log_var (Tensor): Tuple containing the mean and log variance of the latent space.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        return z_mean, z_log_var


class VAE_Decoder_Conv_MNIST(BaseDecoder):
    """
    Convolutional encoder architecture from keras tutorial for a variational autoencoder (VAE).

    Args:
        input_dim (tuple): Dimensions of the input data (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, input_dim, latent_dim):
        BaseDecoder.__init__(self, input_dim, latent_dim)
        self.dense = layers.Dense(7 * 7 * 64, activation="relu")
        self.reshape = layers.Reshape((7, 7, 64))
        self.conv1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv3 = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

    def call(self, z):
        """
        Forward pass.

        Args:
            z (Tensor): Latent space representation.

        Returns:
            x (Tensor): Reconstructed input.
        """
        x = self.dense(z)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


# ------------------------------------------------------------------- #


# --------------------- Recurrent ENCODER-DECODER ------------------- #


class RecurrentEncoder(BaseEncoder):
    """
    Args:
        input_dim (int): Dimensionality of the input data (seq_len, num_features).
        latent_dim (int): Dimensionality of the latent space.
        **kwargs (dict): Additional keyword arguments.

    Attributes:
        masking_value (float): Value for masking sequences.
        mask (Masking): Masking layer.
        h (Bidirectional): Bidirectional LSTM layer.
    """

    def __init__(self, input_dim, latent_dim, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.masking_value = (
            kwargs["masking_value"] if "masking_value" in kwargs else -99.0
        )
        self.mask = layers.Masking(mask_value=self.masking_value)
        self.h = layers.Bidirectional(layers.LSTM(300))
        self.z = layers.Dense(self.latent_dim, name="z")

    def call(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            z (Tensor): Encoder output.
        """
        x = self.mask(x)
        x = self.h(x)
        z = self.z(x)

        return z


class RecurrentDecoder(BaseDecoder):
    """

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
        self.h_decoded_1 = layers.RepeatVector(input_dim[0])
        self.h_decoded_2 = layers.Bidirectional(layers.LSTM(300, return_sequences=True))
        self.h_decoded = layers.LSTM(input_dim[1], return_sequences=True)

    def call(self, z):
        """
        Forward pass.

        Args:
            z (Tensor): Latent space representation.

        Returns:
            x (Tensor): Reconstructed input.
        """
        x = self.h_decoded_1(z)
        x = self.h_decoded_2(x)
        x = self.h_decoded(x)

        return {"x_recon": x}


# ---------------------- SPARSE ENCODER-DECODER --------------------- #
class SparseEncoder(BaseEncoder):
    """
    Sparse Encoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Layers configuration.
        kwargs (dict): Additional keyword arguments.

    Attributes:
        beta (float): Regularization parameter for the sparsity constraint.
        p (float): Target sparsity level for the encoder outputs.
        layers_dict (dict): Dictionary containing the dense layers.
        layers_idx (list): List of layer indices.
        encoder_outputs (Dense): Dense layer with sparsity regularization.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[512], **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim, layers_conf)
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
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Encoder output.
        """
        for _, layer in self.layers_dict.items():
            x = layer(x)
        x = self.encoder_outputs(x)

        return x


class SparseDecoder(BaseDecoder):
    """
    Sparse Decoder class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        layers_conf (list): Layers configuration.

    Attributes:
        layers_dict (dict): Dictionary containing the dense layers of the decoder.
        layers_idx (list): List of layer indices.
        dense_recons (Dense): Dense layer for reconstruction.
    """

    def __init__(self, input_dim, latent_dim, layers_conf=[512], **kwargs):
        BaseDecoder.__init__(self, input_dim, latent_dim, layers_conf)
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
            x (Tensor): Reconstructed input.
        """
        for _, layer in self.layers_dict.items():
            x = layer(x)
        x = self.dense_recons(x)

        return x


class SparseRegularizer(Regularizer):
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
        self.p_hat = ops.mean(x)
        kl_divergence = self.p * (ops.log(self.p / self.p_hat)) + (1 - self.p) * (
            ops.log(1 - self.p / 1 - self.p_hat)
        )

        return self.beta * ops.sum(kl_divergence)

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


# ------------------------ ADDITIONAL MODULES ----------------------- #
class BaseRegressor(layers.Layer):
    """
    Base Regressor class.

    Args:
        units (int): Number of neurons in the dense layer.
        name (str): Name of the regressor layer.

    Attributes:
        dense (Dense): Dense layer in the regressor.
        out (Dense): Output layer in the regressor.
    """

    def __init__(self, units=200, name="regressor"):
        super().__init__(name=name)
        self.dense = layers.Dense(units, activation="tanh")
        self.out = layers.Dense(1, name="reg_output")

    def call(self, x):
        """
        Forward pass.

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
        units (int): Number of neurons in the dense layer.
        name (str): Name of the classifier layer.

    Attributes:
        intermediate (Dense): Intermediate dense layer in the classifier.
        outputs (Dense): Output layer in the classifier.
    """

    def __init__(self, n_classes, units=200, name="classifier"):
        super().__init__(name=name)
        self.intermediate = layers.Dense(units, activation="relu")
        self.outputs = layers.Dense(
            n_classes, activation="softmax", name="class_output"
        )

    def call(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            x (Tensor): Classification output probabilities.
        """
        x = self.intermediate(x)
        x = self.outputs(x)

        return x


# ------------------------------------------------------------------- #
