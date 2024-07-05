import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from sklearn.inspection import DecisionBoundaryDisplay
import jax
import jax.numpy as jnp
from cs349 import *



def convert_categorical_to_int(d, categories):
    r = 0
    for idx, day in enumerate(categories):
        r += (d == day) * idx

    return r


def convert_day_of_week_to_int(d):
    return convert_categorical_to_int(
        d, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    )


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

