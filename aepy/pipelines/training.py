import os
from datetime import datetime
from typing import Optional

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from aepy.models.base import BaseAE
from aepy.pipelines.base import BasePipeline


class TrainingPipeline(BasePipeline):
    """
    Pipeline for end-to-end training of your AE.
    The trained model will be saved in ''output_dir''.
    A folder ''traning_YYY-MM-DD_HH-MM-SS'' will be created inside ''output_dir'',
    where checkpoints and final model will be saved.

    Parameters
    ----------Entonces lo
        model (keras.Model): The model to be trained. Default: None.
        output_dir (str): The directory where the trained model will be saved. Default: None.
        optimizer (str): The optimizer to use for training. Default: Adam.
        learning_rate (float): The learning rate for the optimizer. Default: 0.001.
        num_epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size for training.
    """

    def __init__(
        self,
        name=None,
        model: Optional[BaseAE] = None,
        output_dir="output_dir",
        optimizer="adam",
        learning_rate=0.001,
        batch_size=128,
        num_epochs=100,
    ):
        super().__init__(name, output_dir)
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def __call__(
            self,
            x,
            y=None,
            x_eval=None,
            y_eval=None,
            callbacks: Optional[list] = None,
            save_model: bool = True,):
        """
        Launchs the model training on the provided data

        Args
        ----
            training_data: The training data as a numpy.ndarray of shape (n_samples, n_features).
            eval_data: The evaluation data as a numpy.ndarray of shape (n_samples, n_features).
                       If None, only uses train_data for training. Default: None.
            callbacks: A list of callbacks to use during training. List of keras.callbacks.Callback.
                       Default: 
                       - EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
                       - ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        """
        super().__call__()

        if callbacks is None:
            # set callbacks
            callbacks = []
            if x_eval is None:
                callbacks.append(EarlyStopping(
                    monitor='loss', patience=10, verbose=1, mode='min'))
                callbacks.append(ModelCheckpoint(filepath=os.path.join(self.output_dir, 'model.weights.h5'), monitor='loss',
                                 verbose=1, save_best_only=True, mode='min', save_weights_only=True))
            else:
                callbacks.append(EarlyStopping(
                    monitor='val_loss', patience=10, verbose=1, mode='min'))
                callbacks.append(ModelCheckpoint(filepath=os.path.join(self.output_dir, 'model.weights.h5'), monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min', save_weights_only=True))

        # set optimizer
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        else:
            self.logger.log_error('Unimplemented optimizer')
            exit(-1)

        # compile and fit the model
        self.model.compile(optimizer=optimizer, run_eagerly=True)

        if x_eval is None:
            self.model.fit(
                        x=x,
                        y=y,
                        shuffle=True,
                        epochs=self.num_epochs,
                        batch_size=self.batch_size,
                        callbacks=callbacks,
                        verbose=2)
        else:
            self.model.fit(
                        x=x,
                        y=y,
                        validation_data=(x_eval, y_eval),
                        shuffle=True,
                        epochs=self.num_epochs,
                        batch_size=self.batch_size,
                        callbacks=callbacks,
                        verbose=2)

        # restore the best model
        self.model.load_weights(os.path.join(
            self.output_dir, 'model.weights.h5'))

        if save_model is False:
            os.remove(self.output_dir)

        return self.model
