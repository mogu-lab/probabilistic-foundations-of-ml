import argparse

import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers as H


def convert_categorical_to_int(d, categories):
    r = 0
    for idx, day in enumerate(categories):
        r += (d == day) * idx

    return r


def convert_day_of_week_to_int(d):
    return convert_categorical_to_int(
        d, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    )


def plot_poisson_example():
    S = jnp.arange(40)
    for lamb in [2, 10, 20]:
        prob = jax.scipy.stats.poisson.pmf(S, lamb, loc=0)        
        plt.plot(S, prob)
        plt.scatter(S, prob, label=r'$\lambda = {}$'.format(lamb))

    plt.xlabel(r'$n$')
    plt.ylabel(r'$p_N(n)$')
    plt.title(r'PMF of Poisson Distributions for Different $\lambda$s')
    plt.legend()
    plt.show()


def plot_invariance_of_argmax_under_log():
    x = jnp.linspace(-1.0, 1.0, 100)
    y = -(x ** 2.0) + 1.1

    plt.plot(x, y, label=r'$f(x) = -x^2 + 1$')
    plt.plot(x, jnp.log(y), label=r'$\log f(x)$')
    plt.axvline(x=0.0, label='argmax of $f(x)$ and $\log f(x)$', c='black', ls='--')

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Argmax is invariance to $\log$')
    plt.legend(loc='lower right')
    plt.show()


def cs349_sample(model, key, *args, **kwargs):
    exec = H.trace(H.seed(model, key)).get_trace(*args, **kwargs)
    
    result = dict()
    for k, v in exec.items():
        if v['type'] == 'plate':
            continue

        result[k] = v['value']

    return result


def cs349_mle(model, key, num_steps, *args, learning_rate=0.01, **kwargs):
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

    