from typing import Tuple, Union
from keras import metrics, ops, losses, layers, initializers
from rapidae.models.base import BaseAE


class VQ_VAE(BaseAE):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) model.

    Args:
        beta (float): Hyperparameter for controlling the commitment loss term.
        encoder (BaseEncoder): An instance of BaseEncoder.
        decoder (BaseDecoder): An instance of BaseDecoder.
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        num_embeddings (int): Number of embeddings for the Vector Quantizer.
        layers_conf (list): List specifying the configuration of layers for custom models.
        **kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        beta: float = 0.25,
        latent_dim: int = 2,
        num_embeddings: int = 128,
        encoder: callable = None,
        decoder: callable = None,
        **kwargs,
    ):
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            decoder=decoder,
            **kwargs,
        )

        self.num_embeddings = num_embeddings
        self.beta = beta

        # Create VQ layer
        self.vq_layer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.latent_dim,
            beta=self.beta,
        )

        # Training metrics trackers
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = metrics.Mean(name="vq_loss")

    def call(self, x):
        encoder_outputs = self.encoder(x)
        quantized_latents, vq_loss = self.vq_layer(encoder_outputs)
        vq_loss = ops.array(vq_loss)
        x_recon = self.decoder(encoder_outputs)

        return {
            "quantized_latents": quantized_latents,
            "vq_loss": vq_loss,
            "x_recon": x_recon,
        }

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        recon_loss = ops.mean(losses.mean_squared_error(x, y_pred["x_recon"]))
        self.reconstruction_loss_tracker.update_state(recon_loss)

        # VQ loss was already computed inside the Vector Quantized layer
        vq_loss = y_pred["vq_loss"]
        self.vq_loss_tracker.update_state(vq_loss)

        return recon_loss + vq_loss


class VectorQuantizer(layers.Layer):
    """
    Implements the Vector Quantization layer used in a Vector Quantized Variational Autoencoder (VQ-VAE).
    It quantizes the input data into a set of discrete codes and calculates the commitment loss and codebook loss.

    Args
    ----
        num_embeddings (int): Number of discrete embeddings (codes).
        embedding_dim (int): Dimensionality of each embedding.
        beta (float): Hyperparameter for controlling the commitment loss term.
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper
        self.beta = beta

        # Initialize the embeddings which we will quantize
        w_init = initializers.random_uniform()
        self.embeddings = self.add_weight(
            shape=[self.embedding_dim, self.num_embeddings],
            initializer=w_init,
            name="embeddings_vqvae",
            trainable=True,
            dtype="float32",
        )

    def call(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input data.

        Returns:
            quantized (Tensor), self.losses (Tuple): Tuple containing quantized output and a list of losses.
        """
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact
        input_shape = ops.shape(x)
        flattened = ops.reshape(x, [-1, self.embedding_dim])

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        encodings = ops.one_hot(encoding_indices, self.num_embeddings)
        quantized = ops.matmul(encodings, ops.transpose(self.embeddings))

        # Reshape the quantized values back to the original input shape
        quantized = ops.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer.
        # You can learn more about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/.
        # Check the original paper to get a handle on the formulation of the loss function
        commitment_loss = ops.mean((ops.stop_gradient(quantized) - x) ** 2)
        codebook_loss = ops.mean((quantized - ops.stop_gradient(x)) ** 2)

        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + ops.stop_gradient(quantized - x)

        return quantized, self.losses

    def get_code_indices(self, flattened_inputs):
        """
        Calculate the indices for minimum distances between inputs and codes.

        Args:
        flattened_inputs (tf.Tensor): Flattened input tensor.

        Returns:
        tf.Tensor: Indices for minimum distances.
        """
        # Calculate L2-normalized distance between the inputs and the codes
        similarity = ops.matmul(flattened_inputs, self.embeddings)
        distances = (
            ops.sum(flattened_inputs**2, axis=1, keepdims=True)
            + ops.sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances
        encoding_indices = ops.argmin(distances, axis=1)

        return encoding_indices
