import numpy as np

from sklearn.metrics import accuracy_score
from keras_core import utils
from data.datasets import load_MNIST
from data.utils import evaluate, display_diff, add_noise
from models.ae.ae_model import AE
from models.base.default_architectures import VanillaEncoder, VanillaDecoder
from pipelines.training import TrainingPipeline

# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(persistant=True)

# Obtaint number of clasess
n_classes = len(set(y_train))

# Convert labels to categorical
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)

# Add noise to the train and test data
x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

x_train = x_train.reshape(x_train.shape[0], -1)
x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], -1)

# Display some images to view the added noise
#display_diff(x_train_noisy, x_train)

train_data = dict(data=x_train_noisy.astype(float), labels=x_train)
test_data = dict(data=x_test_noisy.astype(float), labels=y_test)

# Model creation
model = AE(input_dim=(x_train_noisy.shape[0], x_train_noisy.shape[1]), 
            latent_dim=2, encoder=VanillaEncoder, decoder=VanillaDecoder, layers_conf=[64, 32])

pipe = TrainingPipeline(name='training_pipeline',
                        model=model, num_epochs=10)

trained_model = pipe(train_data=train_data)

y_hat = trained_model.predict(test_data)

display_diff(x_test_noisy, y_hat['recon'])

