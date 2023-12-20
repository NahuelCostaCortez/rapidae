from typing import Optional, Union, Tuple

import keras
if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf
elif keras.backend.backend() == 'torch':
    import torch

from aepy.models.base import BaseAE, BaseDecoder, BaseEncoder


class AE(BaseAE):
    """
    Vanilla Autoencoder model.

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
        layers_conf: list = None,
        **kwargs
    ):

        BaseAE.__init__(self, input_dim, latent_dim,
                        encoder=encoder, decoder=decoder, layers_conf=layers_conf)

        # self.decoder = decoder
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    # keras model call function
    def call(self, inputs):
        x = inputs["data"]
        z = self.encoder(x)
        recon_x = self.decoder(z)
        outputs = {}
        outputs["z"] = z
        outputs["recon"] = recon_x
        return outputs

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker, self.total_loss_tracker]

    def compute_losses(self, x, outputs, labels_x=None):
        losses = {}

        losses['recon_loss'] = keras.ops.mean(
            keras.losses.mean_squared_error(x, outputs['recon'])
        )

        return losses

    def update_states(self, losses):
        metrics = {}

        self.reconstruction_loss_tracker.update_state(losses['recon_loss'])
        metrics['reconstruction_loss'] = self.reconstruction_loss_tracker.result()

        self.total_loss_tracker.update_state(losses['total_loss'])
        metrics['total_loss'] = self.total_loss_tracker.result()

        return metrics

    def train_step(self, *args, **kwargs):
        if keras.backend.backend() == 'tensorflow':
            return self._tensorflow_train_step(*args, **kwargs)
        elif keras.backend.backend() == 'torch':
            return self._pytorch_train_step(*args, **kwargs)
    """
    def test_step(self, *args, **kwargs):
        if keras.backend.backend() == 'tensorflow':
            return self._tensorflow_test_step(*args, **kwargs)
        elif keras.backend.backend() == 'pytorch':
            return self._pytorch_test_step(*args, **kwargs)
    """
    def _tensorflow_train_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            losses = self.compute_losses(x, outputs, labels_x)
            #print(losses)
            losses['total_loss'] = sum(losses.values())

        # Compute gradients
        grads = tape.gradient(losses['total_loss'], self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        metrics = self.update_states(losses)

        return metrics
    
    def _pytorch_train_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        outputs = self(inputs)
        losses = self.compute_losses(x, outputs, labels_x)
        #print(losses)
        losses['total_loss'] = sum(losses.values())

        # Zero gradients
        self.zero_grad()

        # Backward pass
        losses['total_loss'].backward()

        # Update weights
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics
        metrics = self.update_states(losses)

        return metrics

    def test_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        outputs = self(inputs)
        losses = self.compute_losses(x, outputs, labels_x)
        losses['total_loss'] = sum(losses.values())

        # Upgrade metrics
        metrics = self.update_states(losses)

        return metrics
