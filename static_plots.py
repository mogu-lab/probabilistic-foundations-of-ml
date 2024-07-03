import os
import math

import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from matplotlib.animation import ArtistAnimation
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro.distributions as D


OUTPUT_DIR = '_static/figs'


def animate_gradient_descent(
    fn, 
    start_x=-2.0, 
    x_domain=(-2.0, 2.0, 0.01), 
    iterations=1000, 
    lr=0.1, 
    precision=None,
    tangent_length=1.0,
    history_length=5,
    figsize=(5, 4),
    annotation_loc=(0.05, 0.7),
    name='anim.gif', 
    fps=5,
):
    '''
    Adapted from: https://www.kaggle.com/code/trolukovich/animating-gradien-descent
    '''

    grad_fn = jax.grad(fn)
    
    images = []    
    fig, ax = plt.subplots(figsize=figsize)
    x = jnp.arange(*x_domain)
    trajectory = [float(start_x)]

    # Function plot
    f = plt.plot(x, fn(x), color='blue', label=r'$\mathcal{L}(\theta)$')

    init = plt.scatter(
        [start_x], [fn(start_x)],
        c='black', marker='*', s=200.0, zorder=3.0,
        label=r'$\theta^{\mathrm{initial}}$',
    )

    for frame in range(iterations):
        # Plot point to track
        fn_at_y = fn(trajectory[-1]) # Y coordinate of point    

        num_points = min(len(trajectory), history_length)
        history = []
        for h in range(num_points):
            plot_cur_legend = (h == num_points - 1 and frame == 1) 
            history.append(plt.scatter(
                trajectory[-num_points + h + 1],
                fn(trajectory[-num_points + h + 1]), 
                color='r',
                zorder=2.5,
                alpha=math.pow((h + 1.0) / float(num_points), 2.0),
                **(dict(label=r'$\theta^{\mathrm{current}}$') if plot_cur_legend else {}),
            ))

        slope = grad_fn(trajectory[-1])
        trajectory.append(trajectory[-1] - lr * slope)
        step = abs(trajectory[-1] - trajectory[-2])

        # Plot text info
        bbox_args = dict(boxstyle='round', fc='0.8')
        text = f'Iteration: {frame}\nPoint: ({trajectory[-2]:.2f}, {fn_at_y:.2f})\nSlope: {slope:.2f}\nStep: {step:.4f}'
        text = ax.annotate(text, xy=(trajectory[-2], fn_at_y), xytext=annotation_loc, textcoords='axes fraction', bbox=bbox_args, fontsize=12)

        plt.title(f'Animation of Gradient Descent: Learning Rate = {lr}')
        plt.legend()

        images.append([f[0], init, text] + history)
        
        # Stopping algorithm if desired precision have been met
        if precision is not None and step <= precision:
            break

    anim = ArtistAnimation(fig, images) 
    anim.save(name, writer='imagemagic', fps=fps)


def all_gradient_descent_plots():
    animate_gradient_descent(
        lambda theta: theta ** 2.0,
        start_x=-2.0, 
        x_domain=(-2.0, 2.0, 0.01), 
        iterations=50, 
        lr=0.1, 
        tangent_length=1.0,
        history_length=20,        
        figsize=(6, 4),
        annotation_loc=(0.34, 0.7),
        name=os.path.join(OUTPUT_DIR, 'gradient_descent_quadratic_fn_lr0p1.gif'), 
        fps=5,
    )

    animate_gradient_descent(
        lambda theta: theta ** 2.0,
        start_x=-2.0, 
        x_domain=(-2.0, 2.0, 0.01), 
        iterations=150, 
        lr=0.01, 
        tangent_length=1.0,
        history_length=20,        
        figsize=(6, 4),
        annotation_loc=(0.34, 0.7),
        name=os.path.join(OUTPUT_DIR, 'gradient_descent_quadratic_fn_lr0p01.gif'), 
        fps=5,
    )        
    
    animate_gradient_descent(
        lambda theta: theta ** 2.0 + jnp.sin(2.0 * jnp.pi * theta),
        start_x=-1.85, 
        x_domain=(-2.0, 2.0, 0.01), 
        iterations=100, 
        lr=0.1, 
        tangent_length=1.0,
        history_length=20,        
        figsize=(6, 4),
        annotation_loc=(0.34, 0.7),
        name=os.path.join(
            OUTPUT_DIR,
            'gradient_descent_quadratic_plus_sin_fn_lr0p1.gif',
        ), 
        fps=5,
    )
    
    animate_gradient_descent(
        lambda theta: theta ** 2.0 + jnp.sin(2.0 * jnp.pi * theta),
        start_x=-1.85, 
        x_domain=(-2.0, 2.0, 0.01), 
        iterations=100, 
        lr=0.01, 
        tangent_length=1.0,
        history_length=20,        
        figsize=(6, 4),
        annotation_loc=(0.34, 0.7),
        name=os.path.join(
            OUTPUT_DIR,
            'gradient_descent_quadratic_plus_sin_fn_lr0p01.gif',
        ), 
        fps=5,
    )


def plot_poisson_example():
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
    S = jnp.arange(40)
    for lamb in [2, 10, 20]:
        prob = jax.scipy.stats.poisson.pmf(S, lamb, loc=0)        
        plt.plot(S, prob)
        plt.scatter(S, prob, label=r'$\lambda = {}$'.format(lamb))

    plt.xlabel(r'$n$')
    plt.ylabel(r'$p_N(n; \lambda)$')
    plt.title(r'PMF of Poisson Distributions for Different $\lambda$s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pmf_of_poisson.png'))
    plt.close()


def plot_invariance_of_argmax_under_log():
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    x = jnp.linspace(-1.0, 1.0, 100)
    y = -(x ** 2.0) + 1.1

    plt.plot(x, y, label=r'$f(x) = -x^2 + 1$')
    plt.plot(x, jnp.log(y), label=r'$\log f(x)$')
    plt.axvline(x=0.0, label='argmax of $f(x)$ and $\log f(x)$', c='black', ls='--')

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('argmax is invariant to $\log$')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'invariance_of_argmax_to_log.png'))
    plt.close()


def plot_example_loss_functions():
    def loss_fn_1(theta):
        return theta ** 2.0
    
    def loss_fn_2(theta):
        return theta ** 2.0 + jnp.sin(2.0 * jnp.pi * theta)
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    theta = jnp.linspace(-2.0, 2.0, 100)
     
    axes[0].plot(theta, loss_fn_1(theta), c='blue', label=r'$\mathcal{L}_1(\theta)$')
    axes[0].axvline(0.0, c='red', ls='--', label='Minimum')
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
    axes[1].axvline(-0.237935, c='red', ls='--', label='Minimum')    
    axes[1].scatter(
        local_optima, 
        loss_fn_2(local_optima), 
        c='black', zorder=1, label=r'$\frac{d \mathcal{L}_2(\theta)}{d \theta} = 0$',
    )
    axes[1].set_xlabel(r'$\theta$')
    axes[1].legend(loc='upper right', framealpha=1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_loss_functions.png'))
    plt.close()



def plot_example_regression():
    def f1(x):
        return 2.0 * x
    
    def f2(x):
        return (5.0 * x ** 4.0) / 6.0 - (10.0 * x ** 2.0) / 3.0 + x / 2.0 + 2.0
    
    key = jrandom.PRNGKey(seed=0)
    key_left, key_right = jrandom.split(key, 2)
    
    p_epsilon = D.Normal(0.0, 0.6)
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    x = jnp.linspace(-2.0, 2.0, 100)

    axes[0].plot(x, f1(x), color='blue', label=r'$\mu(\cdot; W)$')
    axes[0].scatter(
        x, f1(x) + p_epsilon.sample(key_left, x.shape), label=r'Observation Error',
        color='red', alpha=0.5, 
    )

    axes[0].set_title('Linear Regression')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')    
    
    axes[1].plot(x, f2(x), color='blue', label=r'$\mu(\cdot; W)$')
    axes[1].scatter(
        x, f2(x) + p_epsilon.sample(key_right, x.shape), label=r'Observation Error',
        color='red', alpha=0.5,
    )

    axes[1].set_title('Polynomial Regression')    
    axes[1].set_xlabel('$x$')

    for ax in axes:
        leg = ax.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_regression.png'))
    plt.close()
    
    
def main():
    #plot_poisson_example()    
    #plot_invariance_of_argmax_under_log()    
    #plot_example_loss_functions()
    #all_gradient_descent_plots()
    #plot_example_regression()
    pass

    
if __name__ == '__main__':
    main()

