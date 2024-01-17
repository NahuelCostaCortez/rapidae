"""
Class for common model utilities.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import LambdaCallback


class LRFinder:
    """
    Cyclical learning rate finder. The idea is to reduce the amount of guesswork on picking a good starting learning rate.

    Inspired by fastai lr_finder.
    """

    def __init__(self, model):
        """
        Args:
            model: The model object.
        """
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        """
        Callback function called at the end of each batch during training.

        Args:
            batch (int): The index of the current batch.
            logs (dict): Dictionary containing the metrics results for the current batch.
        """
        # Log the learning rate
        lr = self.model.optimizer.learning_rate.value.numpy()
        self.lrs.append(lr)

        # Log the loss
        loss = logs["loss"]
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        self.model.optimizer.learning_rate.assign(lr)

    def find(
        self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1, **kw_fit
    ):
        """
        Finds the optimal learning rate for model training using the LR Range Test.

        Args:
            x_train (numpy array or list of numpy arrays): Training data.
            y_train (numpy array): Target data.
            start_lr (float): Starting learning rate for the LR Range Test.
            end_lr (float): Ending learning rate for the LR Range Test.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 64.
            epochs (int, optional): Number of epochs to train the model. Defaults to 1.
            **kw_fit (dict): Additional keyword arguments to be passed to the `fit` method of the model.
        """
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (
            float(1) / float(num_batches)
        )
        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = self.model.optimizer.learning_rate.value.numpy()

        # Set the initial learning rate
        self.model.optimizer.learning_rate.assign(start_lr)

        callback = LambdaCallback(
            on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs)
        )

        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callback],
            **kw_fit,
        )

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        self.model.optimizer.learning_rate.assign(original_lr)

    def find_generator(
        self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit
    ):
        """
        Finds the optimal learning rate using a generator-based training approach.

        Args:
            generator (data generator): The data generator.
            start_lr (int): The starting learning rate.
            end_lr (int): The ending learning rate.
            epochs (int): The number of epochs to train for. Default is 1.
            steps_per_epoch (int): The number of steps (batches) per epoch. If not specified, it will be inferred from the length of the generator. Default is None.
            **kw_fit (dict): Additional keyword arguments to be passed to the `fit_generator` method.
        """
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(generator)
            except (ValueError, NotImplementedError) as e:
                raise e(
                    "`steps_per_epoch=None` is only valid for a"
                    " generator based on the "
                    "`keras.utils.Sequence`"
                    " class. Please specify `steps_per_epoch` "
                    "or use the `keras.utils.Sequence` class."
                )
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (
            float(1) / float(epochs * steps_per_epoch)
        )

        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = self.model.optimizer.learning_rate.value.numpy()

        # Set the initial learning rate
        self.model.optimizer.learning_rate.assign(start_lr)

        callback = LambdaCallback(
            on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs)
        )

        self.model.fit_generator(
            generator=generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[callback],
            **kw_fit,
        )

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        self.model.optimizer.learning_rate.assign(original_lr)

    def plot_loss(self, n_skip_beginning=1, n_skip_end=1, x_scale="log"):
        """
        Plots the loss through the learning rate range.

        Args:
            n_skip_beginning (int): number of batches to skip on the left.
            n_skip_end (int): number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(
            self.lrs[n_skip_beginning:-n_skip_end],
            self.losses[n_skip_beginning:-n_skip_end],
        )
        plt.xscale(x_scale)
        plt.show()

    def plot_loss_change(
        self, sma=1, n_skip_beginning=1, n_skip_end=1, y_lim=(-0.01, 0.01)
    ):
        """
        Plots rate of change of the loss function.
        Args:
            sma (int): number of batches for simple moving average to smooth out the curve.
            n_skip_beginning (int): number of batches to skip on the left.
            n_skip_end (int): number of batches to skip on the right.
            y_lim (int): limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale("log")
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        """
        Calculate the derivatives of losses using Simple Moving Average (SMA).

        Args:
            sma (int): The window size for calculating the SMA.

        Returns:
            derivatives (list): A list of derivatives calculated using the SMA.
        """
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        """
        Returns the learning rate with the smallest derivative value within a given range.

        Args:
            sma (int): The Simple Moving Average values.
            n_skip_beginning (int): Number of values to skip at the beginning of the derivatives.
            n_skip_end (int): Number of values to skip at the end of the derivatives.

        Returns:
            lrs (float): The learning rate with the smallest derivative value.
        """
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]
