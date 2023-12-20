import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import mean_squared_error
from rapidae.data.datasets import load_MNIST
from rapidae.data.utils import evaluate
from rapidae.models.base.default_architectures import (Decoder_Conv_MNIST, Decoder_MLP,
                                               Encoder_Conv_MNIST, Encoder_MLP)
from rapidae.models.vae.vae_model import VAE
from rapidae.pipelines.training import TrainingPipeline

# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(persistant=True)
train_data = dict(data=x_train.astype(float), labels=y_train)
test_data = dict(data=x_test.astype(float), labels=y_test)

# Model creation
print(x_train.shape[0])
print(x_train.shape[1])
model = VAE(input_dim=(x_train.shape[0], x_train.shape[1]), latent_dim=2, 
            encoder=Encoder_Conv_MNIST, decoder=Decoder_Conv_MNIST, layers_conf=[32, 64])

pipe = TrainingPipeline(name='training_pipeline',
                        model=model, num_epochs=2)

trained_model = pipe(train_data=train_data)
