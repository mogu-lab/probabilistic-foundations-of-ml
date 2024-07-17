import glob
import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from sklearn.inspection import DecisionBoundaryDisplay
import jax
import jax.numpy as jnp

# For loading pre-trained models
import numpyro.distributions.constraints as C
import numpyro.distributions as D
import chex

from cs349 import *


jax.config.update('jax_enable_x64', True)



def convert_categorical_to_int(d, categories):
    r = 0
    for idx, day in enumerate(categories):
        r += (d == day) * idx

    return r


def convert_day_of_week_to_int(d):
    return convert_categorical_to_int(d, [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    ])


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

    samples = cs349_sample_generative_process(
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
        models.append(cs349_load_trained_numpyro_model(fname))

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
    
    samples = cs349_sample_generative_process(
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
