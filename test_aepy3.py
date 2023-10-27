from data.datasets import load_MNIST
from data.utils import evaluate
from models.vanilla_ae.vanilla_ae_model import VanillaAE
from models.base.default_architectures import VanillaEncoder, VanillaDecoder
from pipelines.training import TrainingPipeline


# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(persistant=True)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

train_data = dict(data=x_train.astype(float), labels=y_train)
test_data = dict(data=x_test.astype(float), labels=y_test)

# Model creation
model = VanillaAE(input_dim=(x_train.shape[0], x_train.shape[1]), 
                  latent_dim=2, encoder=VanillaEncoder, decoder=VanillaDecoder, layers_conf=[1024, 512, 256, 64, 32])

pipe = TrainingPipeline(name='pipeline_entrenamiento',
                        model=model, num_epochs=30)

trained_model = pipe(train_data=train_data)
