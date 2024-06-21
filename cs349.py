'''
Helper functions for CS349. 
'''

import argparse

import numpyro
import numpyro.handlers as H


def cs349_sample(model, key, *args, **kwargs):
    '''
    Helper function to sample from a simple numpyro model. 
    The "*args, **kwargs" argument refer to all arguments your model takes in.
    '''
    
    exec = H.trace(H.seed(model, key)).get_trace(*args, **kwargs)
    
    result = dict()
    for k, v in exec.items():
        if v['type'] == 'plate':
            continue

        result[k] = v['value']

    return result


def cs349_mle(model, optimizer, key, num_steps, *args, **kwargs):
    '''
    Helper function to perform MLE on a simple numpyro model.
    The "*args, **kwargs" argument refer to all arguments your model takes in.
    '''

    guide = numpyro.infer.autoguide.AutoDelta(model)
        
    svi = numpyro.infer.SVI(
        model, guide, optimizer, loss=numpyro.infer.Trace_ELBO(),
    )

    svi_result = svi.run(key, num_steps, *args, **kwargs)
    print('Done.')
    
    params = svi_result.params    
    return argparse.Namespace(
        model_mle=H.substitute(model, data=params), 
        parameters_mle=params,
        log_likelihood=-svi_result.losses,
    )

    
