import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error

from data import utils
from data.datasets import load_CMAPSS
from data.preprocessing import CMAPSS_preprocessor
from data.utils import evaluate
from metrics import cmapps_score
from models.base import RecurrentDecoder, RecurrentEncoder
from models.vae import VAE
from pipelines import PreprocessPipeline, TrainingPipeline

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

print(tf.__version__)

# ------------------------------ DATA -----------------------------------
dataset = 'FD003'
# sensors to work with: T30, T50, P30, PS30, phi
sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
# windows length
sequence_length = 30
# smoothing intensity
alpha = 0.1
# max RUL
threshold = 125

# NOT IMPLEMENTED
# x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors,
# sequence_length, alpha, threshold)
train, test, y_test = load_CMAPSS('FD001')
# x_train, y_train, x_val, y_val, x_test, y_test = CMAPSS_preprocessor(train, test, y_test)
preprocess_pipeline = PreprocessPipeline(
    name='CMAPPS_preprocessing', preprocessor=CMAPSS_preprocessor)
x_train, y_train, x_val, y_val, x_test, y_test = preprocess_pipeline(
    train=train, test=test, y_test=y_test, threshold=100)
# -----------------------------------------------------------------------

timesteps = x_train.shape[1]
input_dim = x_train.shape[2]
intermediate_dim = 300
batch_size = 128
latent_dim = 2
epochs = 2
optimizer = 'adam'

# ----------------------------- MODEL -----------------------------------
model = VAE(input_dim=(x_train.shape[1], x_train.shape[2]), latent_dim=2, 
            downstream_task='regression', encoder=RecurrentEncoder, decoder=RecurrentDecoder)
# model_callbacks = utils.get_callbacks("p", model, x_train, y_train)

train_data = dict(data=x_train, labels=y_train)
eval_data = dict(data=x_val, labels=y_val)
pipeline = TrainingPipeline(model=model, num_epochs=1)
trained_model = pipeline(train_data, eval_data)  # , callbacks=model_callbacks)
# -----------------------------------------------------------------------

# ----------------------------- TEST ------------------------------------
test_data = dict(data=x_test, labels=y_test)
y_hat = trained_model.predict(test_data)

evaluate(y_true=np.expand_dims(test_data['labels'], axis=-1),
         y_hat=y_hat['reg'], sel_metric=mean_squared_error)
evaluate(y_true=np.expand_dims(test_data['labels'], axis=-1),
         y_hat=y_hat['reg'], sel_metric=cmapps_score.CMAPSS_Score())
