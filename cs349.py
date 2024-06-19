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


def cs349_mle(model, key, num_steps, *args, learning_rate=0.01, **kwargs):
    '''
    Helper function to perform MLE on a simple numpyro model.
    The "*args, **kwargs" argument refer to all arguments your model takes in.
    '''
    
    def guide(*args, **kwargs):
        pass

    optimizer = numpyro.optim.Adam(step_size=learning_rate)
    svi = numpyro.infer.SVI(
        model, guide, optimizer, loss=numpyro.infer.Trace_ELBO(),
    )

    svi_result = svi.run(key, num_steps, *args, **kwargs)
    params = svi_result.params

    return argparse.Namespace(
        model_mle=H.substitute(model, data=params), 
        parameters_mle=params,
        log_likelihood=-svi_result.losses,
    )

    
