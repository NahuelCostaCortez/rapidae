import numpy as np

from sklearn.metrics import accuracy_score
from keras import utils
from data.datasets import load_MNIST
from data.utils import evaluate, plot_mnist_images
from models.base.default_architectures import (Decoder_Conv_MNIST, Decoder_MLP,
                                               Encoder_Conv_MNIST, Encoder_MLP)
from models.vae.vae_model import VAE
from pipelines.training import TrainingPipeline

# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(persistant=True)

# Obtaint number of clasess
n_classes = len(set(y_train))

# Convert labels to categorical
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)

train_data = dict(data=x_train.astype(float), labels=y_train)
test_data = dict(data=x_test.astype(float), labels=y_test)

# Show some images
plot_mnist_images(x_train, y_train)

# Model creation
model = VAE(input_dim=(x_train.shape[0], x_train.shape[1]), latent_dim=10, downstream_task='classification',
            encoder=Encoder_Conv_MNIST, decoder=Decoder_Conv_MNIST, layers_conf=[32, 64], n_classes=n_classes)

pipe = TrainingPipeline(name='pipeline_entrenamiento',
                        model=model, num_epochs=2)

trained_model = pipe(train_data=train_data)

y_hat = trained_model.predict(test_data)

evaluate(y_true=np.argmax(test_data['labels'], axis=1), 
         y_hat=np.argmax(y_hat['clf'], axis=1),
         sel_metric=accuracy_score)
