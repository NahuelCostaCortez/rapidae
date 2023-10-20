from typing import Optional, Tuple, Union

import tensorflow as tf

from models.base import BaseAE, BaseDecoder, BaseEncoder

from .vanilla_ae_config import VanillaAEConfig


class VanillaAE(BaseAE):
    """
    Vanilla Autoencoder (AE) model.

    Args:
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
    ):

        BaseAE.__init__(self, input_dim, latent_dim,
                        encoder=encoder, decoder=decoder)

        #if self.decoder is not False:
        #    self.decoder = decoder
        #    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
        #        name="reconstruction_loss")

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")

    # keras model call function
    @tf.function
    def call(self, inputs):
        x = inputs['data']
        encoded = self.encoder(x)
        outputs = {}
        outputs['recon'] = self.decoder(encoded)
        #if self.decoder != None:
        #    recon_x = self.decoder(encoded)
        #s    outputs["recon"] = recon_x
        return outputs

    @property
    def metrics(self):
        metrics = [self.reconstruction_loss_tracker]
        if self.decoder != None:
            metrics.append(self.reconstruction_loss_tracker)
        return metrics

    @tf.function
    def train_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]
        with tf.GradientTape() as tape:
            metrics = {}
            encoded = self.encoder(x)
            reconstruction = self.decoder(encoded)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(x, reconstruction)
            )
            metrics['reconstruction_loss'] = reconstruction_loss

            grads = tape.gradient(reconstruction_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            metrics["loss"] = reconstruction_loss 
            
        return metrics

    @tf.function
    def test_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]

        metrics = {}
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.mse(x, reconstruction)
        )
        metrics["reconstruction_loss"] = reconstruction_loss
        metrics["loss"] = reconstruction_loss

        return metrics
