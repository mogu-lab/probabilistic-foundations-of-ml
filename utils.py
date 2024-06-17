import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import jax
import jax.numpy as jnp


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
    