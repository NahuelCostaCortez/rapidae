import sys
from typing import Optional, Tuple, Union

import keras_core as keras
import tensorflow as tf

from aepy.conf import Logger
from aepy.models.base import BaseAE, BaseDecoder, BaseEncoder, BaseRegressor, BaseClassifier


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
                self.weight_vae = kwargs['weight_vae'] if 'weight_vae' in kwargs else 1.0
                self.weight_clf = kwargs['weight_clf'] if 'weight_clf' in kwargs else 1.0
                self.clf_loss_tracker = keras.metrics.Mean(name='clf_loss')
            else:
                Logger().log_warning('The downstream task is not a valid string. Currently there are available "regression" and "classification"')
        else:
            Logger().log_info('No specific dowstream task has been selected')

        if self.decoder is not False:
            # self.decoder = decoder
            self.reconstruction_loss_tracker = keras.metrics.Mean(
                name="reconstruction_loss")

        #self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    # keras model call function
    def call(self, x):
        #x = inputs["data"]
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
    class Sampling(keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a sample."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = keras.ops.shape(z_mean)[0]
            dim = keras.ops.shape(z_mean)[1]
            # added seed for reproducibility
            epsilon = keras.random.normal(shape=(batch, dim), seed=42)
            return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon
    
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # KL loss
        kl_loss = -0.5 * (1 + y_pred['z_log_var'] -
                          keras.ops.square(y_pred['z_mean']) - keras.ops.exp(y_pred['z_log_var']))
        kl_loss = keras.ops.mean(keras.ops.sum(kl_loss, axis=1))
        self.kl_loss_tracker.update_state(kl_loss)

        # Reconstruction loss
        # TODO: Check masking issues
        if self.decoder is not None:
            recon_loss = keras.ops.mean(keras.losses.mean_squared_error(x, y_pred['recon']))
            self.reconstruction_loss_tracker.update_state(recon_loss)
    
        # Regressor lossx
        if self.downstream_task == 'regression':
            reg_loss = keras.ops.mean(keras.losses.mean_squared_error(y, y_pred['reg']))
            self.reg_loss_tracker.update_state(reg_loss)

        # Classifier loss
        if self.downstream_task == 'classification':
            clf_loss = keras.ops.mean(keras.losses.categorical_crossentropy(y, y_pred['clf']))
            self.clf_loss_tracker.update_state(clf_loss)
    
        if self.downstream_task == 'classification':
            if self.decoder is not None:
                loss = self.weight_vae * \
                    (kl_loss + recon_loss) + \
                    self.weight_clf * clf_loss
            else:
                loss = self.weight_vae * \
                    kl_loss + self.weight_clf * \
                    clf_loss
        
        if self.downstream_task == 'regression':
            if self.decoder is not None:
                loss = kl_loss + recon_loss + reg_loss
            else:
                loss = kl_loss + clf_loss
        #self.total_loss_tracker.update_state(loss) # DELETE!

        return loss

"""
class VAECallback(keras.callbacks.Callback):

    def __init__(self, _model, x, y):
        super().__init__()
        self._model = _model
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
    
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

        # KL loss
        kl_loss = -0.5 * (1 + outputs['z_log_var'] -
                          tf.square(outputs['z_mean']) - tf.exp(outputs['z_log_var']))
        losses['kl_loss'] = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        # Regressor loss
        if self.downstream_task == 'regression':
            losses['reg_loss'] = tf.reduce_mean(
                keras.losses.mean_squared_error(labels_x, outputs['reg'])
            )

        # Classifier loss
        if self.downstream_task == 'classification':
            losses['clf_loss'] = tf.reduce_mean(
                keras.losses.categorical_crossentropy(labels_x, outputs['clf'])
            )

        # Reconstruction loss
        # TODO: Check masking issues
        if self.decoder is not None:
            losses['recon_loss'] = tf.reduce_mean(
                keras.losses.mean_squared_error(x, outputs['recon'])
            )

        return losses

    def update_states(self, losses, total_loss):
        metrics = {}

        self.total_loss_tracker.update_state(total_loss)
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

            if self.downstream_task == 'classification':
                if self.decoder is not None:
                    total_loss = self.weight_vae * \
                        (losses['kl_loss'] + losses['recon_loss']) + \
                        self.weight_clf * losses['clf_loss']
                else:
                    total_loss = self.weight_vae * \
                        losses['kl_loss'] + self.weight_clf * \
                        losses['clf_loss']
            else:
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
        if self.downstream_task == 'classification':
            if self.decoder is not None:
                total_loss = self.weight_vae * \
                    (losses['kl_loss'] + losses['recon_loss']) + \
                    self.weight_clf * losses['clf_loss']
            else:
                total_loss = self.weight_vae * \
                    losses['kl_loss'] + self.weight_clf * losses['clf_loss']
        else:
            total_loss = sum(losses.values())

        # Upgrade metrics
        metrics = self.update_states(losses, total_loss)

        return metrics
    """