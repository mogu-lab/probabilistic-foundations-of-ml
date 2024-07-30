import os
import math

import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from matplotlib.animation import ArtistAnimation
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro.distributions as D
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from utils import *
from data import *


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
    N = 100
    
    def f1(x):
        return 2.0 * x
    
    def f2(x):
        return (5.0 * x ** 4.0) / 6.0 - (10.0 * x ** 2.0) / 3.0 + x / 2.0 + 2.0
    
    key = jrandom.PRNGKey(seed=0)
    key_left, key_right = jrandom.split(key, 2)
    
    p_epsilon = D.Normal(0.0, 0.6)
    x = jnp.linspace(-2.0, 2.0, N)
    y1 = f1(x) + p_epsilon.sample(key_left, x.shape)
    y2 = f2(x) + p_epsilon.sample(key_right, x.shape)

    # Plot regression 
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)

    axes[0].plot(x, f1(x), color='blue', label=r'$\mu(\cdot; W)$')
    axes[0].scatter(
        x, y1, label=r'Observation Error',
        color='red', alpha=0.5, 
    )

    axes[0].set_title('Linear Regression')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')    
    
    axes[1].plot(x, f2(x), color='blue', label=r'$\mu(\cdot; W)$')
    axes[1].scatter(
        x, y2, label=r'Observation Error',
        color='red', alpha=0.5,
    )

    axes[1].set_title('Polynomial Regression')    
    axes[1].set_xlabel('$x$')

    for ax in axes:
        leg = ax.legend()
        for lh in leg.legend_handles: 
            lh.set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_regression.png'))
    plt.close()

    # Plot density
    '''
    fig, axes = plt.subplots(
        1, 2, figsize=(7, 3),
        sharex=True, sharey=True, subplot_kw=dict(projection='3d'),
    )
    
    x_grid, y_grid = jnp.meshgrid(
        jnp.linspace(-2.0, 2.0, 200),
        jnp.linspace(-5.0, 5.0, 200),
    )
    
    z_grid = jnp.exp(p_epsilon.log_prob(f1(x_grid) - y_grid))
    s = axes[0].plot_surface(
        x_grid, y_grid, z_grid,
        cmap='viridis',
    )
    
    z_grid = jnp.exp(p_epsilon.log_prob(f2(x_grid) - y_grid))
    s = axes[1].plot_surface(
        x_grid, y_grid, z_grid,
        cmap='viridis',
    )

    for ax in axes:
        ax.view_init(azim=45, elev=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_regression_density.png'))
    plt.close()
    '''

def plot_example_classification():
    # Code source: Gaël Varoquaux
    #              Andreas Müller
    # Modified for documentation by Jaques Grobler
    # License: BSD 3 clause
    # Adapted from:
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    degree = 4
    names = [
        "Linear",
        "Polynomial (Degree={})".format(degree),
        "Neural Network",
    ]

    classifiers = [
        LogisticRegression(max_iter=2000, random_state=42),
        make_pipeline(
            PolynomialFeatures(degree=degree),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
    ]

    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2,
        random_state=1, n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    
    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]

    fig = plt.figure(figsize=(3.0 * len(names), 9.0))
    grid = ImageGrid(
        fig, 111, 
        nrows_ncols=(len(datasets), len(names)),
        axes_pad=0.1,
        aspect=False,
        cbar_mode='single',
        cbar_location='bottom',
        label_mode='L',
        cbar_pad=0.5,
    )

    eps = 0.3
    plot_cbar = True
    
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42,
        )
        
        x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
        y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
        
        cm = plt.cm.binary
        cm_bright = ListedColormap(["red", "cyan"])
        
        # iterate over classifiers
        for col, (name, clf) in enumerate(zip(names, classifiers)):
            ax = grid.axes_row[ds_cnt][col]
            
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            dbd = DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=5.0,
                response_method='predict_proba',
                xlabel=r'$X^{(0)}$', ylabel=r'$X^{(1)}$',
                levels=np.linspace(0.0, 1.0, 11),
            )

            if plot_cbar and 'Neural' in name:
                cbar = grid.cbar_axes[0].colorbar(
                    dbd.surface_,
                    ticks=jnp.linspace(0.0, 1.0, 11, endpoint=True),
                )
                
                cbar.ax.set_xlabel('Probability')
                
                plot_cbar = False
            
            # Plot the data
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.5)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            if ds_cnt == 0:
                ax.set_title(name)

            '''
            # Plot accuracy
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            '''

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_classification.png'))
    plt.close()


def inductive_bias_of_polynomial_regression(percent_ood):
    names = [
        #"Linear",
        "Polynomial (Degree=5)",
        "Polynomial (Degree=6)",
        "Neural Network",
    ]

    classifiers = [
        #LinearRegression(),
        make_pipeline(
            PolynomialFeatures(degree=5),
            LinearRegression(),
        ),
        make_pipeline(
            PolynomialFeatures(degree=6),
            LinearRegression(),
        ),
        MLPRegressor(
            hidden_layer_sizes=(50,),
            max_iter=10000,
            learning_rate='adaptive',
            learning_rate_init=0.01,
            random_state=42,
            n_iter_no_change=100,
        ),
    ]

    df = pd.read_csv(
        'data/IHH-CTR-CGLF-regression-augmented.csv',
        index_col='Patient ID',
    )
    
    datasets = [
        (df['Age'].values[..., None], df['Glow'].values),
        (df['Age'].values[..., None], df['Telekinetic-Ability'].values),
        (df['Glow'].values[..., None], df['Telekinetic-Ability'].values),
    ]

    axis_names = [
        ('Age', 'Glow'),
        ('Age', 'Telekinetic-Ability'),
        ('Glow', 'Telekinetic-Ability'),
    ]

    fig, axes = plt.subplots(
        len(datasets), len(names),
        figsize=(3.0 * len(names), 2.5 * len(datasets)),
    )
    
    # iterate over datasets
    for row, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42,
        )

        eps = (percent_ood / 100.0) * (X.max() - X.min())
        x_min = X.min() #- eps # since these vars cannot be < 0
        x_max = X.max() + eps
        x_test = jnp.linspace(x_min, x_max, 100)[..., None]
        
        # iterate over classifiers
        for col, (name, clf) in enumerate(zip(names, classifiers)):
            if col > 0:
                axes[row, col].sharex(axes[row, 0])
                axes[row, col].sharey(axes[row, 0])
                
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            
            # Plot the data
            axes[row, col].scatter(X, y, c='red', alpha=0.5, label='Data')
            axes[row, col].plot(x_test, clf.predict(x_test), label=r'Trend: $\mu(\cdot; W)$')
            
            axes[row, col].set_xlim(x_min, x_max)

            if row == 0:                
                axes[row, col].set_title(name)                

            if col == 0:
                axes[row, col].set_ylabel(axis_names[row][1])
                
            axes[row, col].set_xlabel(axis_names[row][0])
            axes[row, col].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR,
        'example_regression_inductive_bias_percent_ood_{}.png'.format(percent_ood),
    ))
    plt.close()


def gp_regression_online():
    df = pd.read_csv('data/IHH-CTR-CGLF-regression-augmented.csv')
    X = np.array(df['Age'])
    y = np.array(df['Telekinetic-Ability'])

    X = X[..., None]
    y = y[..., None]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X, y)

    std_dev = ABILITY_STD_DEV.item() / 20.0
    kernel = ConstantKernel(0.1, 'fixed') * RBF() + WhiteKernel(std_dev, 'fixed')
    
    gp_kwargs = dict(
        kernel=kernel,
        random_state=0,
        normalize_y=False,
        n_restarts_optimizer=10,
        alpha=1e-20,
    )
    
    model_full = GaussianProcessRegressor(**gp_kwargs).fit(X, y)

    parameters = model_full.get_params()
    parameters['optimizer'] = None

    margin = 50.0
    test_X = np.linspace(X.min(), X.max() + margin, 200)[..., None]
    
    num_points = [0, 1, 2, 3, 4, 5, 16, 64, 256]
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)
    for idx, (points, ax) in enumerate(zip(num_points, axes.flatten())):
        model = GaussianProcessRegressor(**gp_kwargs).set_params(**parameters)

        if points > 0:
            model = model.fit(X_scaled[:points], y[:points])

        mean, cov = model.predict(scaler.transform(test_X), return_cov=True)
        cov -= np.eye(cov.shape[0]) * (std_dev - 1e-10)

        test_y = jrandom.multivariate_normal(
            jrandom.PRNGKey(seed=0), mean, cov, shape=(30,),
        )

        ax.plot(test_X, test_y.T, color='blue', alpha=0.2, label='Posterior')
        ax.scatter(
            X[:points], y[:points],
            color='black', marker='x', zorder=5, label='Data',
        )
        ax.set_title(r'$N = {}$'.format(points))
        ax.set_ylim([-1.0, 1.0])

        if idx % 3 == 0:
            ax.set_ylabel('Telekinetic-Ability')
        if idx > 9 - 3 - 1:
            ax.set_xlabel('Age')
            

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_online_bayesian_regression.png'))
    plt.close()


def plot_nn_ensemble():
    csv_fname = 'data/IHH-CTR-CGLF-regression-augmented.csv'
    data = pd.read_csv(csv_fname, index_col='Patient ID')

    plt.figure(figsize=(5.5, 3.5))
    
    plt.scatter(
        data['Age'], data['Telekinetic-Ability'],
        color='black', alpha=0.5, marker='x', label='Data',
    )
    plt.axvspan(78, 93, alpha=0.5, color='red', label='No Data')

    X_test = np.linspace(data['Age'].min(), data['Age'].max(), 100)[..., None]
    
    for i in range(10):
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(30, 30, 30 * (i + 1),),
                max_iter=100000,
                learning_rate='adaptive',
                learning_rate_init=0.01,
                random_state=i,
                n_iter_no_change=200,
                alpha=0.0,
            ),
        ).fit(
            data['Age'].values[..., None],
            data['Telekinetic-Ability'].values,
        )

        plt.plot(
            X_test, model.predict(X_test),
            color='blue', alpha=0.5, **(dict(label='Ensemble') if i == 0 else {}),
        )
                
    plt.xlabel('Age')
    plt.ylabel('Telekinetic Ability')
    plt.title('Neural Network Ensemble')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_nn_ensemble_regression.png'))
    plt.close()


def plot_1d_gmm():
    p_x0 = D.Normal(-1.0, 0.2)
    p_x1 = D.Normal(0.0, 0.4)
    p_x2 = D.Normal(2.0, 0.6)

    x = jnp.linspace(-2.0, 4.0, 200)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
    axes[0].plot(x, jnp.exp(p_x0.log_prob(x)), label=r'$\mathcal{N}(\mu_0, \sigma^2_0)$')
    axes[0].plot(x, jnp.exp(p_x1.log_prob(x)), label=r'$\mathcal{N}(\mu_1, \sigma^2_1)$')
    axes[0].plot(x, jnp.exp(p_x2.log_prob(x)), label=r'$\mathcal{N}(\mu_2, \sigma^2_2)$')

    axes[1].plot(
        x,
        (jnp.exp(p_x0.log_prob(x)) + jnp.exp(p_x1.log_prob(x)) + jnp.exp(p_x2.log_prob(x))) / 3.0,
        color='black',
    )

    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('PDF')
    axes[0].set_title('Mixture Components')
    axes[0].legend()

    axes[1].set_xlabel('$x$')
    axes[1].set_title(r'Observed Distribution, $p_X(\cdot; \theta)$')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_1d_gmm.png'))
    plt.close()

    
    
def main():
    #plot_poisson_example()    
    #plot_invariance_of_argmax_under_log()    
    #plot_example_loss_functions()
    #all_gradient_descent_plots()
    #plot_example_regression()
    #plot_example_classification()
    #inductive_bias_of_polynomial_regression(percent_ood=0)
    #inductive_bias_of_polynomial_regression(percent_ood=30)
    #gp_regression_online()
    #plot_nn_ensemble()
    #plot_1d_gmm() 
    pass

    
if __name__ == '__main__':
    main()

