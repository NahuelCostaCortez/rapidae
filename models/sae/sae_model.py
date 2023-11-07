from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf

from conf import Logger
from models.base import BaseAE, BaseDecoder, BaseEncoder


class Sparse_AE(BaseAE):
    """
    Sparse AE
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
        sparsity_target: float = 0.5,
        beta: float = 0.5,
        **kwargs
    ):
    
        BaseAE.__init__(self, input_dim, latent_dim, 
                        encoder=encoder, decoder=decoder, layers_conf=layers_conf)
        
        if sparsity_target < 0.0:
            Logger().log_error('Sparsity_target cannot be less than 0')

        #if sparsity_weight < 0.0:
        #    Logger().log_error('Sparsity_weight cannot be less than 0')

        self.sparsity_target = sparsity_target
        self.beta = beta
        #self.sparsity_weight = sparsity_weight
        self.reg_layer = tf.keras.layers.ActivityRegularization(l1=self.sparsity_target)

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.penalty_loss_tracker = tf.keras.metrics.Mean(name="penalty")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    
    # keras model call function
    def call(self, inputs):
        x = inputs["data"]
        z = self.encoder(x)
        z = self.reg_layer(z)
        recon_x = self.decoder(z)
        outputs = {}
        outputs["z"] = z
        outputs["recon"] = recon_x
        return outputs
    
    @property
    def metrics(self):
        metrics = [self.reconstruction_loss_tracker, self.penalty_loss_tracker]
        return metrics
    
    @tf.function
    def train_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]
        with tf.GradientTape() as tape:
            metrics = {}

            z = self.encoder(x)
            z = self.reg_layer(z)  # Apply sparsity regularization
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(x, reconstruction)
            )
            metrics["reconstruction_loss"] = reconstruction_loss
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)

            # Compute mean activation values and sparsity regularization
            rho_hat = tf.reduce_mean(z, axis=0)
            print(rho_hat)
            kl_loss = self.beta * tf.reduce_sum(
                self.sparsity_target * tf.math.log(self.sparsity_target / rho_hat) +
                (1 - self.sparsity_target) * tf.math.log((1 - self.sparsity_target) / (1 - rho_hat))
            )
            metrics["kl_loss"] = kl_loss
            self.penalty_loss_tracker.update_state(kl_loss)  # Update penalty loss

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        metrics["loss"] = total_loss

        return metrics

    
    @tf.function
    def test_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]

        metrics = {}

        z = self.encoder(x)
        z = self.reg_layer(z)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(labels_x, reconstruction)
        )
        metrics["loss"] = reconstruction_loss

        return metrics