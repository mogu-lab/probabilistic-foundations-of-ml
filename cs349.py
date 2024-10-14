###############################################################################
# Functions for Fitting NumPyro Models in CS349. DO NOT EDIT.
# Version 0
###############################################################################


import argparse
import dill

import jax
import jax.random as jrandom
import jax.numpy as jnp
import numpyro
import numpyro.handlers as H


def cs349_sample_generative_process(model, key, *args, num_samples=None, **kwargs):
    '''
    A function to sample from a simple numpyro model.

    Arguments:
        model: a function representing a numpyro model
        key: a random generator key
        num_samples: number of samples to draw from the model
        *args, **kwargs: captures all arguments your model takes in

    Note: 
        What are *args and **kwargs? See this tutorial:
        https://www.digitalocean.com/community/tutorials/how-to-use-args-and-kwargs-in-python-3
        
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

    if num_samples is None:
        return sample_once(key)
        
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

    Note: 
        What are *args and **kwargs? See this tutorial:
        https://www.digitalocean.com/community/tutorials/how-to-use-args-and-kwargs-in-python-3
    
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


def cs349_bayesian_inference(model, key, num_warmup, num_samples, *args, **kwargs):
    '''
    A function that performs Bayesian inference on a given model

    Arguments:
        model: a function representing a numpyro model
        key: controls the randomness of the sampler
        num_warmup: number of iterations used for warming up the algorithm
        num_samples: number of samples to return
        *args, **kwargs: captures all arguments your model takes in

    Note: 
        What are *args and **kwargs? See this tutorial:
        https://www.digitalocean.com/community/tutorials/how-to-use-args-and-kwargs-in-python-3
    
    Returns:
        Posterior samples
    '''
    
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            model,
            max_tree_depth=10,
            target_accept_prob=0.6,
        ), 
        num_warmup=num_warmup, 
        num_samples=num_samples,
    )
    
    mcmc.run(key, *args, **kwargs)
    samples = mcmc.get_samples()

    return samples


def cs349_sample_predictive(model, key, posterior_samples, *args, **kwargs):
    '''
    A function that feeds samples from the posterior through the model again, useful for predictions

    Arguments:
        model: a function representing a numpyro model
        key: controls the randomness of the sampler
        posterior_samles: samples from the posterior
        *args, **kwargs: captures all arguments your model takes in

    Note: 
        What are *args and **kwargs? See this tutorial:
        https://www.digitalocean.com/community/tutorials/how-to-use-args-and-kwargs-in-python-3

    Returns:
        The model's joint data log-likelihood
    '''
    
    predictive = numpyro.infer.util.Predictive(
        model, 
        posterior_samples=posterior_samples,
        exclude_deterministic=True,
    )
    
    return predictive(key, *args, **kwargs)
    

def cs349_joint_data_log_likelihood(model, *args, **kwargs):
    '''
    A function that computes a fitted model's joint data log-likelihood

    Arguments:
        model: a function representing a numpyro model
        *args, **kwargs: captures all arguments your model takes in

    Note: 
        What are *args and **kwargs? See this tutorial:
        https://www.digitalocean.com/community/tutorials/how-to-use-args-and-kwargs-in-python-3
    
    Returns:
        The model's joint data log-likelihood
    '''
    return numpyro.infer.util.log_density(model, args, kwargs, {})[0]
    


def cs349_save_trained_numpyro_model(model, parameters, fname):
    '''
    A function to save a numpyro model to a file

    Arguments:
        model: a function representing a numpyro model
        parameters: a dictionary containing parameters of the fitted model
        fname: name of file for storing the model
    
    Returns: Nothing.
    '''
    with open(fname, 'wb') as f:
        dill.dump(dict(
            model=model,
            parameters=parameters,
        ), f)

        
def cs349_load_trained_numpyro_model(fname):
    '''
    A function to load a numpyro model from a file

    Arguments:
        fname: name of file for storing the model
    
    Returns: the model
    '''
    
    with open(fname, 'rb') as f:        
        r = dill.load(f)

    return H.substitute(r['model'], data=r['parameters'])


def cs349_mle_continuous_lvm(model, optimizer, key, num_steps, *args, **kwargs):
    '''
    A function to perform MLE on a numpyro model with continuous latent variables.
    Note: this function assumes NO DISCRETE latent variables.

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

    guide = numpyro.infer.autoguide.AutoNormal(model)
        
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


'''
# IGNORE!
def cs349_mle_discrete_lvm(model, optimizer, key, num_steps, *args, latent_variables=[], **kwargs):
    key_init, key_opt = jrandom.split(key, 2)
    
    global_model = H.block(
        H.seed(
            F.config_enumerate(model), 
            key_init,
        ), 
        hide=latent_variables,
    )

    global_guide = numpyro.infer.autoguide.AutoDelta(global_model)
        
    svi = numpyro.infer.SVI(
        global_model, global_guide, optimizer, loss=numpyro.infer.TraceEnum_ELBO(),
        )
    
        svi_result = svi.run(key_opt, num_steps, *args, **kwargs)
        print('Done.')
    
    params = svi_result.params    
    result = argparse.Namespace(
        model_mle=H.substitute(model, data=params), 
        parameters_mle=params,
        losses=svi_result.losses,
        log_likelihood=-svi_result.losses,        
    )

    return result
'''
