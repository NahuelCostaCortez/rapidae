import sys
from typing import Optional, Tuple, Union

import keras
import tensorflow as tf

from aepy.conf import Logger
from aepy.models.base import BaseAE, BaseDecoder, BaseEncoder, BaseRegressor, BaseClassifier


class VQ_VAE(BaseAE):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) model.

    Args:
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        num_embeddings: int = 128,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
        **kwargs
    ):
        
        BaseAE.__init__(self, input_dim, latent_dim,
                encoder=encoder, decoder=decoder, layers_conf=layers_conf)
        
        self.num_embeddings = num_embeddings
        
        # Create VQ layer
        self.vq_layer = VectorQuantizer(num_embeddings=self.num_embeddings, 
                                        embedding_dim=self.latent_dim)

        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')

    def call(self, inputs):
        x = inputs['data']
        encoder_outputs = self.encoder(x)
        print(encoder_outputs.shape)
        quantized_latents, vq_loss = self.vq_layer(encoder_outputs)
        outputs={}
        outputs['vq_loss'] = vq_loss
        outputs['quantized_latents'] = quantized_latents
        recon_x = self.decoder(encoder_outputs)
        outputs['recon'] = recon_x

        return outputs

    @property
    def metrics(self):
        metrics = [self.total_loss_tracker,
                   self.reconstruction_loss_tracker,
                   self.vq_loss_tracker]
        return metrics
    
    def compute_losses(self, x, outputs, labels_x=None):
        losses = {}

        losses['recon_loss'] = keras.ops.mean(
                    keras.losses.mean_squared_error(x, outputs['recon'])
                )
        # VQ loss was already computed in Vector Quantized layer
        losses['vq_loss'] = outputs['vq_loss']
        
        return losses

    def update_states(self, losses, total_loss):
        metrics = {}
        
        self.total_loss_tracker.update_state(total_loss)
        metrics['loss'] = self.total_loss_tracker.result()

        self.reconstruction_loss_tracker.update_state(losses['recon_loss'])
        metrics['recontruction_loss'] = self.reconstruction_loss_tracker.result()

        self.vq_loss_tracker.update_state(losses['vq_loss'])
        metrics['vq_loss'] = self.vq_loss_tracker.result()

        return metrics
    
    @tf.function
    def train_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            losses = self.compute_losses(x, outputs, labels_x)
            total_loss = sum(losses.values())

        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        metrics = self.update_states(losses, total_loss)

        return metrics

    @tf.function
    def test_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        outputs = self(inputs)
        losses = self.compute_losses(x, outputs, labels_x)
        total_loss = sum(losses.values())

        # Update metrics
        metrics = self.update_states(losses, total_loss)

        return metrics

class VectorQuantizer(keras.layers.Layer):
    """
    implements the Vector Quantization layer used in a Vector Quantized Variational Autoencoder (VQ-VAE).
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

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = keras.initializers.random_uniform()
        #w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact
        input_shape = keras.ops.shape(x)
        flattened = keras.ops.reshape(x, [-1, self.embedding_dim])

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        encodings = keras.ops.one_hot(encoding_indices, self.num_embeddings)
        quantized = keras.ops.matmul(encodings, keras.ops.transpose(self.embeddings))

        # Reshape the quantized values back to the original input shape
        quantized = keras.ops.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function
        commitment_loss = keras.ops.mean((keras.ops.stop_gradient(quantized) - x) ** 2)
        codebook_loss = keras.ops.mean((quantized - keras.ops.stop_gradient(x)) ** 2)

        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + keras.ops.stop_gradient(quantized - x)

        return quantized, self.losses

    def get_code_indices(self, flattened_inputs):
        """
        Calculate the indices for minimum distances between inputs and codes.

        Args:
        flattened_inputs (tf.Tensor): Flattened input tensor.

        Returns:
        tf.Tensor: Indices for minimum distances.
        """
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = keras.ops.matmul(flattened_inputs, self.embeddings)
        distances = (
            keras.ops.sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + keras.ops.sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = keras.ops.argmin(distances, axis=1)
        
        return encoding_indices
