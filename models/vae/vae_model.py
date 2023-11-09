from typing import Optional, Tuple, Union

import tensorflow as tf

from conf import Logger
from models.base import BaseAE, BaseDecoder, BaseEncoder, BaseRegressor, BaseClassifier


class VAE(BaseAE):
    """
    Variational Autoencoder (VAE) model.

    Args:
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        masking_value: float = -99.0,
        exclude_decoder: bool = False,
        downstream_task: str = None,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
        **kwargs
    ):
        
        BaseAE.__init__(self, input_dim, latent_dim,
                        encoder=encoder, decoder=decoder, masking_value=masking_value, layers_conf=layers_conf)
        
        self.downstream_task = downstream_task
        if self.downstream_task is not None:
            # Change string to lowercase
            self.downstream_task = downstream_task.lower()
            if self.downstream_task == 'regression':
                Logger().log_info('Regressor available for the latent space of the autoencoder')
                self.regressor = BaseRegressor()
                self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
            elif self.downstream_task == 'classification':
                Logger().log_info('Classificator available for the latent space of the autoencoder')
                self.classifier = BaseClassifier(kwargs['n_classes'])
                self.clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
            else:
                Logger().log_warning('The downstream task is not a valid string. Currently there are available "regression" and "classification"')
        else:
            Logger().log_info('No specific dowstream task has been selected')

        if self.decoder is not False:
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
        if self.downstream_task == 'regression':
            reg_prediction = self.regressor(z)
            outputs["reg"] = reg_prediction
        if self.downstream_task == 'classification':
            clf_prediction = self.classifier(z)
            outputs["clf"] = clf_prediction
        if self.decoder != None:
            recon_x = self.decoder(z)
            outputs["recon"] = recon_x
        return outputs

    # TODO: cambiar a una función que se llame NormalSampler
    class Sampling(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a sample."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            # added seed for reproducibility
            epsilon = tf.random.normal(shape=(batch, dim), seed=42)
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @property
    def metrics(self):
        metrics = [self.total_loss_tracker, self.kl_loss_tracker]
        if self.decoder != None:
            metrics.append(self.reconstruction_loss_tracker)
        if self.downstream_task == 'regression':
            metrics.append(self.reg_loss_tracker)
        if self.downstream_task == 'classification':
            metrics.append(self.clf_loss_tracker)
        return metrics

    def compute_losses(self, x, outputs, labels_x=None):
        # KL divergence loss
        kl_loss = -0.5 * (
            1 + outputs['z_log_var'] -  tf.square(outputs['z_mean']) - tf.exp(outputs['z_log_var'])
        )
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        # Regressor loss
        reg_loss = 0
        if self.downstream_task == 'regression':
            reg_loss = tf.reduce_mean(
                    tf.keras.losses.mse(labels_x, outputs['reg'])
                )
        
        # Classifier loss
        clf_loss = 0
        if self.downstream_task == 'classification':
            clf_loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(labels_x, outputs['clf'])
                )
        
        # Reconstruction loss
        recon_loss = 0
        if self.decoder is not None:
            recon_loss = tf.reduce_mean(
                    tf.keras.losses.mse(x, outputs['recon']), axis=1
                )
        
        return kl_loss, recon_loss, reg_loss, clf_loss

    def update_states(self, total_loss, kl_loss, recon_loss=None, reg_loss=None, clf_loss=None):
        metrics = {}

        self.total_loss_tracker.update_state(total_loss)
        metrics['total_loss'] = self.total_loss_tracker.result()

        self.kl_loss_tracker.update_state(kl_loss)
        metrics['kl_loss'] = self.kl_loss_tracker.result()

        if self.downstream_task == 'regression':
            self.reg_loss_tracker.update_state(reg_loss)
            metrics['reg_loss'] = self.reg_loss_tracker.result()
        if self.downstream_task == 'classification':
            self.clf_loss_tracker.update_state(clf_loss)
            metrics['clf_loss'] = self.clf_loss_tracker.result()
        if self.decoder is not None:
            self.reconstruction_loss_tracker.update_state(recon_loss)
            metrics['reconstruction_loss'] = self.reconstruction_loss_tracker.result()
        
        return metrics
        
    @tf.function
    def train_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            kl_loss, recon_loss, reg_loss, clf_loss = self.compute_losses(x, outputs, labels_x)
            total_loss = kl_loss + recon_loss + reg_loss + clf_loss
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        metrics = self.update_states(total_loss, recon_loss, reg_loss, clf_loss)

        return metrics

    @tf.function
    def test_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]

        # Forward pass
        outputs = self(inputs)
        kl_loss, recon_loss, reg_loss, clf_loss = self.compute_losses(x, outputs, labels_x)
        total_loss = kl_loss + recon_loss + reg_loss + clf_loss

        # Upgrade metrics
        metrics = self.update_states(total_loss, recon_loss, reg_loss, clf_loss)

        return metrics
