import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras import utils

from rapidae.data.datasets import load_MNIST
from rapidae.data.utils import evaluate, display_diff, add_noise
from rapidae.models.vq_vae.vq_vae_model import VQ_VAE
from rapidae.models.base.default_architectures import Encoder_Conv_VQ_MNIST, Decoder_Conv_VQ_MNIST
from rapidae.pipelines.training import TrainingPipeline

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

print(tf.__version__)

# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(persistant=True)

# Obtaint number of clasess
n_classes = len(set(y_train))

# Convert labels to categorical
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)

train_data = dict(data=x_train.astype(float), labels=x_train)
test_data = dict(data=x_test.astype(float), labels=y_test)

# Model creation
model = VQ_VAE(input_dim=(x_train.shape[0], x_train.shape[1]), num_embeddings=64,
            latent_dim=2, encoder=Encoder_Conv_VQ_MNIST, decoder=Decoder_Conv_VQ_MNIST, layers_conf=[32, 64])

pipe = TrainingPipeline(name='training_pipeline',
                        model=model, num_epochs=10)

trained_model = pipe(train_data)

y_hat = trained_model.predict(test_data)

display_diff(x_test, y_hat['recon'])