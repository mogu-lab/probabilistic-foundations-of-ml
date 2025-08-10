###############################################################################
# Helper Functions for Assignments. DO NOT EDIT.
# Version 0
###############################################################################

import math
import glob

import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.animation
from IPython.display import HTML
from sklearn.inspection import DecisionBoundaryDisplay
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.example_libraries.stax as stax

# For loading pre-trained models
import numpyro.distributions.constraints as C
import numpyro.distributions as D
import chex

import probabilistic_foundations_of_ml as pfml


jax.config.update('jax_enable_x64', True)


###############################################################################
# Utiliites for Units on Directed Graphical Models
###############################################################################

DAYS_OF_WEEK = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
]

CONDITIONS = [
    'Entangled Antennas', 'Allergic Reaction', 'Intoxication', 'High Fever', 'Broken Limb', 
]

def convert_categorical_to_int(cs, categories):
    r = 0
    for idx, c in enumerate(categories):
        r += (cs == c) * idx

    return r


def convert_day_of_week_to_int(d):
    '''
    For use in the joint probability unit: converts days of the week to integers
    '''
    return convert_categorical_to_int(d, DAYS_OF_WEEK)


def convert_condition_to_int(c):
    '''
    For use in the joint probability unit: converts conditions to integers
    '''
    return convert_categorical_to_int(c, CONDITIONS)


###############################################################################
# Utiliites for Units on Predictive Models
###############################################################################


def plot_classifier_of_control_vs_age_and_dose(
    data, 
    fitted_model, 
    control_sample_site, 
    num_samples=500, 
    num_grid=500,
):
    '''
    For use in the classification unit: plots your classifier against the data

    Arguments:
        data: the data set, passed in as a pandas DataFrame
        fitted_model: your fitted numpyro model
        control_sample_site: the string passed into the numpyro.sample function for sampling control
        num_samples: number of samples drawn to compute the probability
        num_grid: number of points when drawing the shaded region

    Return:
        Nothing
    '''
    
    test_age, test_dose = jnp.meshgrid(
        jnp.linspace(0.0, 100.0, num_grid),
        jnp.linspace(0.0, 1.0, num_grid),
    )

    samples = pfml.sample_generative_process(
        fitted_model, 
        jrandom.PRNGKey(seed=0), 
        num_grid ** 2.0, 
        test_age.reshape(-1),
        test_dose.reshape(-1),
        num_samples=num_samples,
    )

    p_control_given_age_and_dose = (
        samples[control_sample_site].mean(axis=0)
    ).reshape(test_age.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    prob_levels = jnp.linspace(0.0, 1.0, 11, endpoint=True)
    dbd = DecisionBoundaryDisplay(
        xx0=test_age, 
        xx1=test_dose, 
        response=p_control_given_age_and_dose,        
    ).plot(
        ax=ax,
        xlabel='Age',
        ylabel='Dose',
        cmap=plt.cm.binary,
        levels=prob_levels,
    )
    
    in_control = (data['Control-After'] == 1)
    ax.scatter(
        data['Age'][in_control], data['Dose'][in_control], 
        alpha=0.5, s=10.0, c='cyan', label='In Control',
    )
    ax.scatter(
        data['Age'][~in_control], data['Dose'][~in_control], 
        alpha=0.5, s=10.0, c='red', label='Not In Control',
    )
    
    ax.set_xlim(test_age.min(), test_age.max())
    ax.set_ylim(test_dose.min(), test_dose.max())
    
    ax.set_title('Telekinetic Control vs. Age and Dose')
    leg = ax.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    
    cbar = fig.colorbar(dbd.surface_, ticks=prob_levels)
    cbar.ax.set_ylabel('Probability of Telekinetic Control')
    
    plt.tight_layout()
    plt.show()


def load_all_regression_models_of_comfort_vs_intensity():
    models = []
    for fname in sorted(glob.glob('data/regression_model_eval_metrics_*.dill')):
        models.append(pfml.load_trained_numpyro_model(fname))

    return models
    

def plot_regression_model_of_comfort_vs_intensity(
        data, 
        fitted_model,
        ax,
):
    '''
    For use in the evaluation metrics unit: plot the regression models against the data

    Arguments:
        data: the data set, passed in as a pandas DataFrame
        fitted_model: your fitted numpyro model
        ax: a matplotlib axis object on which to plot

    Return:
        Nothing
    '''
    key = jrandom.PRNGKey(seed=0)
    
    test_x = jnp.linspace(
        data['Intensity'].min(), data['Intensity'].max(), 200,
    )[..., None]
    
    samples = pfml.sample_generative_process(
        fitted_model, key, len(test_x), test_x,
    )

    ax.fill_between(
        test_x.squeeze(),
        samples['mu'].squeeze() - samples['std_dev'] * 2.0,
        samples['mu'].squeeze() + samples['std_dev'] * 2.0,
        color='blue', alpha=0.2, label=r'$95\%$ of Samples',
    )
    ax.scatter(
        data['Intensity'], data['Comfort'],
        c='black', marker='x', alpha=0.8, label='Data', 
    )    
    ax.plot(
        test_x.squeeze(), samples['mu'],
        c='red', alpha=0.8, label=r'$\mu(\cdot; W)$',
    )
        
    ax.legend(loc='lower right')


def plot_all_regression_models_of_comfort_vs_intensity(data, models):
    '''
    For use in the evaluation metrics unit: plot ALL regression models against the data

    Arguments:
        data: the data set, passed in as a pandas DataFrame
        models: a list of fitted numpyro models

    Return:
        Nothing
    '''
    
    rows = 2
    cols = 3
    assert(rows * cols <= len(models))
    
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 3.5 * rows), sharex=True, 
    )
    
    for idx, (ax, m) in enumerate(zip(axes.flatten(), models)):
        plot_regression_model_of_comfort_vs_intensity(data, m, ax)

        ax.set_title('Model {}'.format(idx))
        if idx % cols == 0:
            ax.set_ylabel('Comfort')
        if idx > rows * cols - cols - 1:
            ax.set_xlabel('Intensity')

    plt.tight_layout()
    plt.show()


###############################################################################
# Utiliites for Units on Generative Models
###############################################################################


def neural_network_fn(name, N, layers, activation_fn=stax.LeakyRelu):
    '''
    For use in the factor analysis unit: creates a neural network function

    Arguments:
        name: name of neural network (should be unique for each numpyro model)
        N: number of observations (usually this is the size of the plate)
        layers: a list representing the number of hidden layers in the network
                e.g. [2, 50, 100, 5] is a neural network that transform 
                2-dimensional inputs into 5-dimensional outputs using two hidden layers, 
                one with 50 and one with 100 neurons.
        activation_fn: choice of activation function

    Return:
        A neural network function for use in a numpyro model. 
        In calling this function, the network's parameters will be 
        automatically created and initialized:

        fn = neural_network_fn('NN', 100, [2, 50, 100, 5])
        
        The network can then be used as follows:

        y = fn(x)        
    '''
    
    modules = []
    for idx, out_dim in enumerate(layers[1:]):
        # Create a linear transform
        # In deep learning lingo, this is called a "dense layer"
        modules.append(stax.Dense(
            out_dim, W_init=stax.randn(),
        ))

        # Create an activation function
        if idx < len(layers) - 2:
            modules.append(activation_fn)

    # Register the neural networks parameters with numpyro.param
    return numpyro.module(
        name,
        stax.serial(*modules),
        input_shape=(N, layers[0]),
    )
        

def visualize_microscope_samples(samples):
    '''
    For use in the factor analysis unit: visualizes samples

    Arguments:
        samples: An (S, 24 * 24)-shaped array of generated samples
                 where S is the number of samples

    Return:
        Nothing
    '''
    
    chex.assert_rank(samples, 2)
    im_width = int(math.sqrt(samples.shape[-1]))
    
    chex.assert_axis_dimension(samples, -1, im_width * im_width)

    grid_width = math.ceil(math.sqrt(samples.shape[0]))

    fig = plt.figure(figsize=(grid_width, grid_width))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(grid_width, grid_width),  
        axes_pad=0.0,
        share_all=True,
    )

    grid[0].get_xaxis().set_ticks([])    
    grid[0].get_yaxis().set_ticks([])
    
    for ax, im in zip(grid, samples):
        ax.imshow(im.reshape(im_width, im_width), cmap='gray', vmin=0.0, vmax=1.0)

    plt.show()


def animate_latent_space_path(result, z1, z2, num_points=50, speed=10):
    '''
    For use in the factor analysis unit: animates a path in the latent space

    Arguments:
        result: what's returned by the function, pfml.mle_continuous_lvm
        z1: a starting coordinate in the latent space 
        z2: an ending coordinate in the latent space
        num_points: number of points connecting z1 to z2
        speed: number of frames-per-second in the generated animation

    Return:
        HTML code that will be rendered in a Jupyter notebook. 
    '''
    assert(z1.shape == (2,))
    assert(z2.shape == (2,))
    
    if 'z_auto_loc' not in result.parameters_mle:
        raise Exception('Make sure your latent variable is named "z", and that you use the appropriate MLE function for this model')
    
    # Get the 'z' most likely to have generated every 'x'
    z_mu = result.parameters_mle['z_auto_loc']
    assert(z_mu.ndim == 2)
    assert(z_mu.shape[-1] == 2)

    # Interpolate between z1 to z2 to create a "path" in the latent space
    rho = jnp.linspace(0.0, 1.0, num_points)[..., None]
    z_path = z1[None, ...] * rho + z2[None, ...] * (1.0 - rho)

    # Decode the latent variables along the path
    path = pfml.sample_generative_process(
        H.condition(result.model_mle, data=dict(z=z_path)), 
        jrandom.PRNGKey(seed=0), 
        N=z_path.shape[0], 
    )

    if 'mu' not in path:
        raise Exception('You must name the output of the decoder function "mu" using numpy.deterministic')
    
    x_path = path['mu']
    assert(x_path.shape == (num_points, 24 * 24))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    axes[0].scatter(z_mu[:, 0], z_mu[:, 1], alpha=0.2)
    axes[0].plot(z_path[[0, -1], 0], z_path[[0, -1], 1], color='black')
    axes[0].set_title('Data Projected into Latent Space')    
    cur = axes[0].scatter(z_path[0:1, 0], z_path[0:1, 1], color='red', marker='*', s=70, zorder=5)

    axes[1].set_title('Corresponding Image')
    
    def animate(i):
        cur.set_offsets(z_path[i:(i+1)])
        axes[1].imshow(x_path[i].reshape(24, 24), cmap='gray', vmin=0.0, vmax=1.0)
    
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(x_path))        
    html = HTML(ani.to_jshtml(default_mode='reflect', fps=speed))
    plt.close()
    
    return html
    
