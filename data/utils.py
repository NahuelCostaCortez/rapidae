"""
Class for common data utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def viz_latent_space(exp_name, model, data, targets=[], epoch='Final', save=False, show=False, path='./'):
    z = model.call(dict(data=data))['z']
    
    plt.figure(figsize=(8, 10))
    if len(targets)>0:
        plt.scatter(z[:, 0], z[:, 1], c=targets)
    else:
        plt.scatter(z[:, 0], z[:, 1])
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    if show:
        plt.show()
    if save:
        plt.savefig(path+exp_name+'_latent_space.png')
    return z

"""
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
"""