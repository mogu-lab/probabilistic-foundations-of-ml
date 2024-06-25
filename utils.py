import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import jax
import jax.numpy as jnp


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
    plt.title('argmax is invariant to $\log$')
    plt.legend(loc='lower right')
    plt.show()


def plot_example_loss_functions():
    def loss_fn_1(theta):
        return theta ** 2.0
    
    def loss_fn_2(theta):
        return theta ** 2.0 + jnp.sin(2.0 * jnp.pi * theta)
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    theta = jnp.linspace(-2.0, 2.0, 100)
     
    axes[0].plot(theta, loss_fn_1(theta), c='blue', label=r'$\mathcal{L}_1(\theta)$')
    axes[0].axvline(0.0, c='red', ls='--', label='Minima')
    axes[0].scatter(
        jnp.zeros(1), 
        loss_fn_1(jnp.zeros(1)), 
        c='black', zorder=1, label=r'$\frac{d \mathcal{L}_1(\theta)}{d \theta} = 0$',
    )
    axes[0].set_xlabel(r'$\theta$')
    axes[0].set_ylabel(r'$\mathcal{L}(\theta)$')
    axes[0].legend(loc='upper right', framealpha=1.0)

    local_optima = jnp.array(
        [-1.85023, -1.18827, -0.790481, -0.237935, 0.263358, 0.713534, 1.31896, 1.66132]
    )
    axes[1].plot(theta, loss_fn_2(theta), c='blue', label=r'$\mathcal{L}_2(\theta)$')
    axes[1].axvline(-0.237935, c='red', ls='--', label='Minima')    
    axes[1].scatter(
        local_optima, 
        loss_fn_2(local_optima), 
        c='black', zorder=1, label=r'$\frac{d \mathcal{L}_2(\theta)}{d \theta} = 0$',
    )
    axes[1].set_xlabel(r'$\theta$')
    axes[1].legend(loc='upper right', framealpha=1.0)
    
    plt.tight_layout()
    plt.show()

