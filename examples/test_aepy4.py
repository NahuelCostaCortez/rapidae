import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from keras import utils
from rapidae.data.datasets import load_MNIST
from rapidae.data.utils import evaluate
from rapidae.models.base.default_architectures import Decoder_Conv_MNIST, Encoder_Conv_MNIST
from rapidae.models.vae.vae_model import VAE
from rapidae.pipelines.training import TrainingPipeline

# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(persistant=True)

# Obtaint number of clasess
n_classes = len(set(y_train))

# Convert labels to categorical
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)

#x_train = np.expand_dims(x_train, -1).astype("float32") / 255
#x_test = np.expand_dims(x_test, -1).astype("float32") / 255

train_data = dict(data=x_train.astype(float), labels=y_train)
test_data = dict(data=x_test.astype(float), labels=y_test)

print(x_train.shape)
# Model creation
model = VAE(input_dim=(x_train.shape[0], x_train.shape[1]), latent_dim=2, downstream_task='classification',
            encoder=Encoder_Conv_MNIST, decoder=Decoder_Conv_MNIST, layers_conf=[32, 64], n_classes=n_classes)

pipe = TrainingPipeline(name='training_pipeline',
                        model=model, num_epochs=2)

trained_model = pipe(train_data=train_data)

y_hat = trained_model.predict(test_data)

evaluate(y_true=np.argmax(test_data['labels'], axis=1), 
         y_hat=np.argmax(y_hat['clf'], axis=1),
         sel_metric=accuracy_score)

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test, np.argmax(y_hat, axis=1), target_names=target_names))
