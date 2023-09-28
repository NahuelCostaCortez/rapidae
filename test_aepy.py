from models.base.default_architectures import Decoder_Conv_MNIST, Encoder_Conv_MNIST, Encoder_MLP, Decoder_MLP
from models.vae.vae_model import VAE, VAEConfig
from pipelines.training import TrainingPipeline
from data.datasets import load_MNIST


# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(use_keras=True)
train_data = dict(data=x_train.astype(float), labels=y_train)
test_data = dict(data=x_test.astype(float), labels=y_test)

# Model creation
model_config = VAEConfig(input_dim=(x_train.shape[0], x_train.shape[1]), latent_dim=2)
model = VAE(model_config=model_config, encoder=Encoder_Conv_MNIST(model_config), decoder=Decoder_Conv_MNIST(model_config))

pipe = TrainingPipeline(name='pipeline_entrenamiento', model=model, num_epochs=2)

trained_model = pipe(train_data=train_data)

y_hat = model.predict(test_data)