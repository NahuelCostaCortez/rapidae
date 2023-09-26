from models.base.default_architectures import Decoder_Conv_MNIST, Encoder_Conv_MNIST, Encoder_MLP, Decoder_MLP
from models.ae.ae_model import AE, AEConfig
from pipelines.training import TrainingPipeline
from data.datasets import load_MNIST


# Load MNIST dataset
x_train, y_train, x_test, y_test = load_MNIST(use_keras=True)
train_data = dict(data=x_train.astype(float), labels=y_train)
eval_data = dict(data=x_test.astype(float), labels=y_test)

# Model creation
model_config = AEConfig(input_dim=(x_train.shape[0], x_train.shape[1]), latent_dim=2)
model = AE(model_config=model_config, encoder=Encoder_Conv_MNIST(model_config),  decoder=Decoder_Conv_MNIST(model_config))

pipe = TrainingPipeline(model=model, num_epochs=10)

trained_model = pipe(train_data=train_data, eval_data=eval_data)


