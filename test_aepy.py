from sklearn.metrics import mean_squared_error
from data.datasets import load_MNIST
from data.utils import evaluate
from models.base.default_architectures import (Decoder_Conv_MNIST, Decoder_MLP,
                                               Encoder_Conv_MNIST, Encoder_MLP)
from models.vae.vae_model import VAE
from pipelines.training import TrainingPipeline

# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(persistant=False)
train_data = dict(data=x_train.astype(float), labels=y_train)
test_data = dict(data=x_test.astype(float), labels=y_test)

# Model creation
#model_config = VAEConfig(input_dim=(
#    x_train.shape[0], x_train.shape[1]), latent_dim=2)
model = VAE(input_dim=(x_train.shape[0], x_train.shape[1]), latent_dim=2, 
            encoder=Encoder_Conv_MNIST, decoder=Decoder_Conv_MNIST)

pipe = TrainingPipeline(name='pipeline_entrenamiento',
                        model=model, num_epochs=2)

trained_model = pipe(train_data=train_data)
