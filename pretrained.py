import os

import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pandas as pd
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import numpyro
import numpyro.distributions as D
import numpyro.distributions.constraints as C
import numpyro.handlers as H
import chex
import jax.example_libraries.stax as stax

from utils import *
from cs349 import *


VIZ_DIR = '_static/figs'


#########################################################################
# IHH Center for Space Immunology: Data and Models for Evaluation Metrics
#########################################################################


def generate_regression_eval_metrics_data():
    def helper(N, seed, set_name):
        key = jrandom.PRNGKey(seed=seed)
        key_x, key_z, key_eps = jrandom.split(key, 3)
        
        x = (1.5 + jax.random.truncated_normal(
            key_x,
            -1.5,
            3.0,
            shape=(N,),
        )) / 4.5
        
        z = jax.random.bernoulli(key_z, p=0.15, shape=(N,))
        
        mu = jnp.where(
            z,
            jnp.polyval(5.0 * jnp.array([-1.0, 1.0, 0.0, 0.04]), x),
            jnp.polyval(5.0 * jnp.array([-1.0, 0.7, 0.3, 0.04]), x),
        )
        
        eps = jax.random.normal(key_eps, shape=mu.shape) * 0.1
        y = mu + eps
        
        df = pd.DataFrame({
            'Magnitude': x,
            'Resistance': y,
        })
        
        df.index.name = 'Patient ID'
        df.to_csv(os.path.join('data', 'IHH-CSI-{}.csv'.format(set_name)))

    helper(140, 0, 'train')
    helper(40, 1, 'val')
    helper(20, 2, 'test')


def stax_nn(layers, activation_fn=stax.LeakyRelu):
    modules = []
    for idx, out_dim in enumerate(layers[1:]):
        modules.append(stax.Dense(out_dim, W_init=stax.randn()))

        if idx < len(layers) - 2:
            modules.append(activation_fn)

    return stax.serial(*modules)


def neural_network_regression_univariate(
        N, x, y=None, hidden_layers=[], activation_fn=stax.LeakyRelu,
):
    chex.assert_rank(x, 2)
    chex.assert_axis_dimension(x, -1, 1)
    if y is not None:
        chex.assert_rank(y, 2)
        chex.assert_axis_dimension(y, -1, 1)        
        
    fn = numpyro.module(
        'NN',
        stax_nn([1] + hidden_layers + [1], activation_fn=activation_fn),
        input_shape=x.shape,
    )

    std_dev = numpyro.param(
        'std_dev',
        0.1,
        constraint=C.positive,
    )
    
    with numpyro.plate('data', N, dim=-2):
        mu = numpyro.deterministic('mu', fn(x))
        p_y_given_x = D.Normal(mu, std_dev)
        numpyro.sample('y', p_y_given_x, obs=y)

    
def fit_and_save_regression_eval_metrics_models():    
    data = pd.read_csv(
        os.path.join('data', 'IHH-CSI-train.csv'),
        index_col='Patient ID',
    )
    
    NUM_ITERATIONS = 50000
    
    def fit_helper(model, key):
        optimizer = numpyro.optim.Adam(step_size=0.01)
    
        result = cs349_mle(
            model,
            optimizer, 
            key, 
            NUM_ITERATIONS,
            len(data), 
            jnp.array(data['Magnitude'])[..., None], 
            y=jnp.array(data['Resistance'])[..., None],
        )

        return result

    model_type1 = lambda N, x, y=None: neural_network_regression_univariate(
        N, x, y=y, hidden_layers=[],
    )

    model_type2 = lambda N, x, y=None: neural_network_regression_univariate(
        N, x, y=y, hidden_layers=[50], activation_fn=stax.Sigmoid,
    )
    
    model_type3 = lambda N, x, y=None: neural_network_regression_univariate(
        N, x, y=y, hidden_layers=[50, 50, 50],
    )
    
    models = [model_type1, model_type2, model_type3]
    names = ['Underfitting', 'Just Right', 'Overfitting']
    
    key = jrandom.PRNGKey(seed=0)
    fitted_models = []
    for model in models:
        fitted_models.append(fit_helper(model, key))
        
    fig, axes = plt.subplots(
        1, len(models), figsize=(4 * len(models), 4), sharex=True, 
    )
    
    for idx, (ax, result, name) in enumerate(zip(axes, fitted_models, names)):
        plot_regression_of_resistance_vs_magnitude(
            data, 
            result.model_mle,
            ax,
        )

        ax.set_title(name)           
        ax.set_xlabel('Magnitude')
        if idx == 0:
            ax.set_ylabel('Resistance')

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'eval_metrics_nn.png'))
    plt.close()
    

def main():
    generate_regression_eval_metrics_data()
    fit_and_save_regression_eval_metrics_models()    


if __name__ == '__main__':
    main()

