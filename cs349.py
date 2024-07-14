'''
Helper functions for CS349. 
'''

import argparse
import dill

import jax
import jax.random as jrandom
import jax.numpy as jnp
import numpyro
import numpyro.handlers as H


def cs349_sample_generative_process(model, key, *args, num_samples=1, **kwargs):
    '''
    A function to sample from a simple numpyro model.

    Arguments:
        model: a function representing a numpyro model
        key: a random generator key
        num_samples: number of samples to draw from the model
        *args, **kwargs: captures all arguments your model takes in
        
    Returns: A dictionary of samples    
    '''

    def sample_once(key):
        exec = H.trace(H.seed(model, key)).get_trace(*args, **kwargs)
        
        result = dict()
        for k, v in exec.items():
            if v['type'] == 'plate':
                continue
    
            result[k] = v['value']
    
        return result

    return jax.vmap(sample_once)(jrandom.split(key, num_samples))


def cs349_mle(model, optimizer, key, num_steps, *args, **kwargs):
    '''
    A function to perform MLE on a simple numpyro model.

    Arguments:
        model: a function representing a numpyro model
        optimizer: a numpyro optimizer (e.g. optimizer = numpyro.optim.Adam(step_size=0.01))
        key: a random generator key
        num_steps: the number of iterations of gradient descent 
        *args, **kwargs: captures all arguments your model takes in
    
    Returns: An object containing,
        model_mle: The model with parameters fixed those that maximize the likelihood
        parameters: The values of the parameters that maximize the likelihood
        losses: the loss function for the MLE for every step of gradient descent
        log_likelihood: the log-likelihood of the model for every step of gradient descent
    '''

    guide = numpyro.infer.autoguide.AutoDelta(model)
        
    svi = numpyro.infer.SVI(
        model, guide, optimizer, loss=numpyro.infer.Trace_ELBO(),
    )

    svi_result = svi.run(key, num_steps, *args, **kwargs)
    print('Done.')
    
    params = svi_result.params    
    result = argparse.Namespace(
        model_mle=H.substitute(model, data=params), 
        parameters_mle=params,
        losses=svi_result.losses,
        log_likelihood=-svi_result.losses,        
    )

    return result

    
def cs349_save_trained_numpyro_model(model, parameters, fname):
    with open(fname, 'wb') as f:
        dill.dump(dict(
            model=model,
            parameters=parameters,
        ), f)

        
def cs349_load_trained_numpyro_model(fname):
    with open(fname, 'rb') as f:        
        r = dill.load(f)

    return H.substitute(r['model'], data=r['parameters'])
    
