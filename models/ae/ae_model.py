from typing import Optional, Union, Tuple

import tensorflow as tf

from models.base import BaseAE, BaseDecoder, BaseEncoder


class AE(BaseAE):
    """
    Vanilla Autoencoder model.

    Args
    ----
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None
    ):

        BaseAE.__init__(self, input_dim, latent_dim,
                        encoder=encoder, decoder=decoder, layers_conf=layers_conf)

        #self.decoder = decoder
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    # keras model call function
    @tf.function
    def call(self, inputs):
        x = inputs["data"]
        z_mean, z_log_var = self.encoder(x)
        z = self.Sampling()([z_mean, z_log_var])
        outputs = {}
        outputs["z"] = z
        outputs["z_mean"] = z_mean
        outputs["z_log_var"] = z_log_var
        recon_x = self.decoder(z)
        outputs["recon"] = recon_x
        return outputs

    class Sampling(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim), seed=42)
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker]

    @tf.function
    def train_step(self, inputs):
        x = inputs["data"]
        with tf.GradientTape() as tape:
            metrics = {}

            z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(x, reconstruction)
            )
            metrics["reconstruction_loss"] = reconstruction_loss

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return metrics

    @tf.function
    def test_step(self, inputs):
        x = inputs[0]["data"]
        labels_x = inputs[0]["labels"]

        metrics = {}

        z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x, reconstruction)
        )
        metrics["reconstruction_loss"] = reconstruction_loss

        return metrics
