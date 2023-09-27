from data import utils 
from data.preprocessing import CMAPSS_preprocessor
from data.datasets import load_CMAPSS
from models.base import RecurrentEncoder, RecurrentDecoder
from models.vae import VAEConfig, VAE
from metrics import mean_squared_error, r2_score
from pipelines import TrainingPipeline

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
#x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors, 
#sequence_length, alpha, threshold)
train, test, y_test =  load_CMAPSS('FD001')
x_train, y_train, x_val, y_val, x_test, y_test = CMAPSS_preprocessor(train, test, y_test)
# -----------------------------------------------------------------------

timesteps = x_train.shape[1]
input_dim = x_train.shape[2]
intermediate_dim = 300
batch_size = 128
latent_dim = 2
epochs = 2
optimizer = 'adam'

# ----------------------------- MODEL -----------------------------------
model_config = VAEConfig(input_dim=(x_train.shape[1],x_train.shape[2]), latent_dim=2, exclude_decoder=True, regressor=True)
model = VAE(model_config=model_config, encoder=RecurrentEncoder(model_config), decoder=RecurrentDecoder(model_config))
#model_callbacks = utils.get_callbacks("p", model, x_train, y_train)

train_data = dict(data=x_train, labels=y_train)
eval_data = dict(data=x_val, labels=y_val)
pipeline = TrainingPipeline(model=model, num_epochs=1)
trained_model = pipeline(train_data, eval_data)#, callbacks=model_callbacks)
# -----------------------------------------------------------------------

# ----------------------------- TEST ------------------------------------
test_data = dict(data=x_test, labels=y_test)
y_hat = trained_model.predict(test_data)
mse = mean_squared_error.MeanSquaredError()
results = mse.calculate(y_test, y_hat)
# -----------------------------------------------------------------------