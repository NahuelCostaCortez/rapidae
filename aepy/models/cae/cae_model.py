from typing import Optional, Union, Tuple

import keras_core as keras
import tensorflow as tf

from aepy.models.base import BaseAE, BaseDecoder, BaseEncoder


class CAE(BaseAE):
    """
    Contractive Autoencoder model.

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

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name='reconstruction_loss')
        self.contractive_loss_tracker = keras.metrics.Mean(
            name='contractive_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')

    def call(self, inputs):
        x = inputs['data']
        x_hid = self.encoder(x)
        recon_x = self.decoder(x_hid)
        outputs = {}
        outputs['x_hidden'] = x_hid
        outputs['recon'] = recon_x
        print(x.shape)
        print(recon_x.shape)
        return outputs

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker,
                self.contractive_loss_tracker,
                self.total_loss_tracker]

    def compute_losses(self, x, outputs, labels_x=None):
        losses = {}

        # Reconstruction loss
        losses['recon_loss'] = tf.reduce_mean(
            keras.losses.mean_squared_error(x, outputs['recon'])
        )

        # Contractive loss
        lambda_ = 1e-4
        n_layers = len(self.encoder.layers_dict)
        last_layer_name = f'dense_{n_layers - 1}'
        last_layer = self.encoder.layers_dict.get(last_layer_name)

        W = last_layer.weights[0]  # N x N_hidden
        W = tf.transpose(W)  # N_hidden x N
        h = outputs['x_hidden']
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = lambda_ * \
            tf.reduce_sum(tf.linalg.matmul(dh**2, tf.square(W)), axis=1)

        losses['contractive_loss'] = contractive_loss

        return losses

    def update_states(self, losses):
        metrics = {}

        self.reconstruction_loss_tracker.update_state(losses['recon_loss'])
        metrics['reconstruction_loss'] = self.reconstruction_loss_tracker.result()

        self.contractive_loss_tracker.update_state(losses['contractive_loss'])
        metrics['contractive_loss'] = self.contractive_loss_tracker.result()

        self.total_loss_tracker.update_state(losses['total_loss'])
        metrics['total_loss'] = self.total_loss_tracker.result()

        return metrics

    @tf.function
    def train_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            losses = self.compute_losses(x, outputs, labels_x)
            # print(losses)
            losses['total_loss'] = sum(losses.values())

        # Compute gradients
        grads = tape.gradient(losses['total_loss'], self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        metrics = self.update_states(losses)

        return metrics

    @tf.function
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
