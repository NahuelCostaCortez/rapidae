from typing import Optional, Tuple, Union

import keras_core as keras
import tensorflow as tf

from keras_core import layers
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
                self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")
            elif self.downstream_task == 'classification':
                Logger().log_info('Classificator available for the latent space of the autoencoder')
                self.classifier = BaseClassifier(kwargs['n_classes'])
                self.clf_loss_tracker = keras.metrics.Mean(name='clf_loss')
            else:
                Logger().log_warning('The downstream task is not a valid string. Currently there are available "regression" and "classification"')
        else:
            Logger().log_info('No specific dowstream task has been selected')

        if self.decoder is not False:
            #self.decoder = decoder
            self.reconstruction_loss_tracker = keras.metrics.Mean(
                name="reconstruction_loss")

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    # keras model call function
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
    class Sampling(layers.Layer):
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
        losses = {}

        # KL divergence loss
        kl_loss = -0.5 * (
            1 + outputs['z_log_var'] -  tf.square(outputs['z_mean']) - tf.exp(outputs['z_log_var'])
        )
        losses['kl_loss'] = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        print("KL Loss:", losses['kl_loss'])

        # Regressor loss
        if self.downstream_task == 'regression':
            losses['reg_loss'] = tf.reduce_mean(
                    keras.losses.mean_squared_error(labels_x, outputs['reg'])
                )
            print("REG Loss:", losses['reg_loss'])
        
        # Classifier loss
        if self.downstream_task == 'classification':
            losses['clf_loss'] = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(labels_x, outputs['clf'])
                )
            print("CLF Loss:", losses['clf_loss'])
        
        # Reconstruction loss
        if self.decoder is not None:
            losses['recon_loss'] = tf.reduce_mean(
                    keras.losses.mean_squared_error(x, outputs['recon'])
                )
            print("RECON Loss:", losses['recon_loss'])
        
        return losses
    
    def update_states(self, losses):
        metrics = {}

        self.total_loss_tracker.update_state(losses['total_loss'])
        metrics['total_loss'] = self.total_loss_tracker.result()

        self.kl_loss_tracker.update_state(losses['kl_loss'])
        metrics['kl_loss'] = self.kl_loss_tracker.result()

        if self.downstream_task == 'regression':
            self.reg_loss_tracker.update_state(losses['reg_loss'])
            metrics['reg_loss'] = self.reg_loss_tracker.result()
        if self.downstream_task == 'classification':
            self.clf_loss_tracker.update_state(losses['clf_loss'])
            metrics['clf_loss'] = self.clf_loss_tracker.result()
        if self.decoder is not None:
            self.reconstruction_loss_tracker.update_state(losses['recon_loss'])
            metrics['reconstruction_loss'] = self.reconstruction_loss_tracker.result()
        
        return metrics
    
    @tf.function    
    def train_step(self, inputs):
        x = inputs['data']
        labels_x = inputs['labels']

        # Forward pass
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            losses = self.compute_losses(x, outputs, labels_x)
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
        x = inputs["data"]
        labels_x = inputs["labels"]

        # Forward pass
        outputs = self(inputs)
        losses = self.compute_losses(x, outputs, labels_x)
        losses['total_loss'] = sum(losses.values())

        # Upgrade metrics
        metrics = self.update_states(losses)

        return metrics
