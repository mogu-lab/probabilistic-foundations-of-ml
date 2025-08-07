import copy
import os

import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro

from utils import *
import probabilistic_foundations_of_ml as pfml


DATA_DIR = 'data'
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
            'Intensity': x,
            'Comfort': y,
            'Race': np.array(['Ezakiens', 'Thadori'])[z.astype('int32')],
        })
        
        df.index.name = 'Patient ID'
        df.to_csv(os.path.join('data', 'IHH-CRD-{}-augmented.csv'.format(set_name)))

        df.drop(columns=['Race']).to_csv(
            os.path.join('data', 'IHH-CRD-{}.csv'.format(set_name)),
        )
        
    helper(140, 0, 'train')
    helper(40, 1, 'val')
    helper(20, 2, 'test')


def fit_and_save_regression_eval_metrics_models():    
    data = pd.read_csv(
        os.path.join('data', 'IHH-CRD-train.csv'),
        index_col='Patient ID',
    )
    
    NUM_ITERATIONS = 50000
    
    def fit_helper(model, key):
        optimizer = numpyro.optim.Adam(step_size=0.01)
    
        result = pfml.mle(
            model,
            optimizer, 
            key, 
            NUM_ITERATIONS,
            len(data), 
            jnp.array(data['Intensity'])[..., None], 
            y=jnp.array(data['Comfort'])[..., None],
        )

        return result

    def stax_nn(layers, activation_fn=jax.example_libraries.stax.LeakyRelu):
        modules = []
        for idx, out_dim in enumerate(layers[1:]):
            modules.append(jax.example_libraries.stax.Dense(
                out_dim, W_init=jax.example_libraries.stax.randn(),
            ))
            
            if idx < len(layers) - 2:
                modules.append(activation_fn)

        return jax.example_libraries.stax.serial(*modules)

    def neural_network_regression_univariate(
            N, x, y=None, hidden_layers=[],
            activation_fn=jax.example_libraries.stax.LeakyRelu,
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

    model0 = lambda N, x, y=None: neural_network_regression_univariate(
        N, x, y=y, hidden_layers=[],
    )
    
    model1 = lambda N, x, y=None: neural_network_regression_univariate(
        N, x, y=y, hidden_layers=[200],
        activation_fn=jax.example_libraries.stax.Sigmoid,
    )
    
    model2 = lambda N, x, y=None: neural_network_regression_univariate(
        N, x, y=y, hidden_layers=[50, 50, 40],
    )
    
    models = [model0, model1, model2]
    
    key = jrandom.PRNGKey(seed=0)
    results = []
    for idx, model in enumerate(models):
        result = fit_helper(model, key)
        results.append(result)
        
        pfml.save_trained_numpyro_model(
            result.model_mle,
            result.parameters_mle,
            os.path.join(
                DATA_DIR,
                'regression_model_eval_metrics_{}.dill'.format(idx),
            ),
        )


    r = results[1]
    
    parameters = copy.deepcopy(r.parameters_mle)    
    parameters['std_dev'] *= 2.0

    pfml.save_trained_numpyro_model(
        model1,
        parameters,
        os.path.join(
            DATA_DIR,
            'regression_model_eval_metrics_{}.dill'.format(len(models)),
        ),
    )

    parameters = copy.deepcopy(r.parameters_mle)    
    parameters['std_dev'] /= 2.0
    
    pfml.save_trained_numpyro_model(
        model1,
        parameters,
        os.path.join(
            DATA_DIR,
            'regression_model_eval_metrics_{}.dill'.format(len(models) + 1),
        ),
    )

    parameters = copy.deepcopy(r.parameters_mle)
    parameters['std_dev'] *= 2.0

    # The long way of setting parameters['NN$params'][2][1] = 0.3    
    parameters = jax.tree.map(
        lambda x: 0.3 if (x == parameters['NN$params'][2][1]).all() else x,
        parameters,
    )
    
    pfml.save_trained_numpyro_model(
        model1,
        parameters,
        os.path.join(
            DATA_DIR,
            'regression_model_eval_metrics_{}.dill'.format(len(models) + 2),
        ),
    )
    
    
def main():
    #generate_regression_eval_metrics_data()
    #fit_and_save_regression_eval_metrics_models()
    pass


if __name__ == '__main__':
    main()

