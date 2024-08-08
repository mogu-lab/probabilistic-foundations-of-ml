import argparse
from collections import defaultdict
import os
import glob

from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pandas as pd
import sklearn.datasets
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import numpyro
import numpyro.distributions as D
import numpyro.distributions.constraints as C
import numpyro.handlers as H
import chex


DATA_DIR = 'data'


def jax_int_array_to_str_list(a, idx_to_s):
    return [idx_to_s[i] for i in a.tolist()]


#########################################################################
# IHH ER Discrete Data
#########################################################################

IDX_TO_DAY_OF_WEEK = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]


IDX_TO_CONDITION = [
    'High Fever',    
    'Broken Limb',    
    'Entangled Antennas',
    'Allergic Reaction',
    'Intoxication',
]

IDX_TO_BOOL = [
    'No',
    'Yes',
]


def model_discrete_IHH_ER(N, data=None):
    assert(N > 0)
    if data is None:
        data = {
            'Day-of-Week': None,
            'Condition': None,
            'Hospitalized': None,
            'Antibiotics': None,
            'Knots': None,
        }
    
    pi_day_of_week = numpyro.param(
        'pi_day_of_week',
        jnp.array([0.20, 0.15, 0.14, 0.14, 0.14, 0.11, 0.12]),
        constraint=C.simplex,
    )

    pi_condition_given_is_weekend = numpyro.param(
        'pi_condition_given_is_weekend',
        jnp.array([
            [0.3, 0.1, 0.1, 0.4, 0.1],
            [0.2, 0.06666667, 0.06666667, 0.26666667, 0.4],
        ]),
        constraint=C.simplex,        
    )

    pi_hospitalized_given_condition = numpyro.param(
        'pi_hospitalized_given_condition',
        jnp.array([0.7, 0.1, 0.0, 0.5, 0.1]),
        constraint=C.unit_interval,
    )

    pi_antibiotics_given_is_allergy_and_hospitalized = numpyro.param(
        'pi_antibiotics_given_condition_and_hospitalized',
        jnp.array([0.0, 0.8]),
        constraint=C.unit_interval,        
    )

    lambda_knots = numpyro.param(
        'lambda_knots',
        jnp.array([0.0, 3.0]),
        constraint=C.positive,
    )

    with numpyro.plate('data', N):
        p_day_of_week = D.Categorical(pi_day_of_week)                    
        day_of_week = numpyro.sample(
            'Day-of-Week', p_day_of_week, obs=data['Day-of-Week'],
        )
        chex.assert_shape(day_of_week, (N,))

        is_weekend = ((day_of_week % 7 == 5) | (day_of_week % 7 == 6)).astype('int32')
        p_condition_given_is_weekend = D.Categorical(
            pi_condition_given_is_weekend[is_weekend],
        )
        condition = numpyro.sample(
            'Condition', p_condition_given_is_weekend, obs=data['Condition'],
        )
        chex.assert_shape(condition, (N,))
        
        p_hospitalized_given_condition = D.Bernoulli(
            pi_hospitalized_given_condition[condition],
        )
        hospitalized = numpyro.sample(
            'Hospitalized', p_hospitalized_given_condition, obs=data['Hospitalized'],
        )
        chex.assert_shape(hospitalized, (N,))

        allergy_and_hospitalized = ((condition == 3) & hospitalized).astype('int32')
        p_antibiotics = D.Bernoulli(
            pi_antibiotics_given_is_allergy_and_hospitalized[allergy_and_hospitalized]
        )
        antibiotics = numpyro.sample(
            'Antibiotics', p_antibiotics, obs=data['Antibiotics'],
        )
        chex.assert_shape(antibiotics, (N,))

        entangled = (condition == 2).astype('int32')
        p_knots = D.Poisson(
            lambda_knots[entangled],
        )
        knots = numpyro.sample(
            'Knots',
            p_knots,
            obs=data['Knots']
        )
        chex.assert_shape(knots, (N,))


def generate_IHH_ER_data_discrete():
    N = 10000
    with H.seed(rng_seed=0):
        exec_trace = H.trace(model_discrete_IHH_ER).get_trace(N)

    df = pd.DataFrame({
        'Day-of-Week': jax_int_array_to_str_list(
            exec_trace['Day-of-Week']['value'],
            IDX_TO_DAY_OF_WEEK,
        ),
        'Condition': jax_int_array_to_str_list(
            exec_trace['Condition']['value'],
            IDX_TO_CONDITION,
        ),
        'Hospitalized': jax_int_array_to_str_list(
            exec_trace['Hospitalized']['value'],
            IDX_TO_BOOL,
        ),
        'Antibiotics': jax_int_array_to_str_list(
            exec_trace['Antibiotics']['value'],
            IDX_TO_BOOL,
        ),
        'Knots': (
            exec_trace['Knots']['value']
        ),
    })

    df.index.name = 'Patient ID'
    df.to_csv(os.path.join(DATA_DIR, 'IHH-ER.csv'))


#########################################################################
# IHH Telekinesis Center: Discrete-Continuous Data
#########################################################################


IDX_TO_UNDERLYING_CONDITION = ['Entangled Antennas', 'Allergic Reaction', 'Intoxication']


def model_discete_continuous_IHH_CTR(N, c=None, a=None):
    pi = numpyro.param(
        'pi',
        jnp.array([0.2, 0.3, 0.5]),
        constraint=C.simplex,
    )

    means = numpyro.param(
        'pi',
        jnp.array([-2.0, 0.0, 2.0]),
        constraint=C.real,
    )

    std_devs = numpyro.param(
        'pi',
        jnp.array([0.2, 0.6, 0.4]),
        constraint=C.positive,
    )
    
    with numpyro.plate('data', N):
        p_C = D.Categorical(pi)
        c = numpyro.sample('c', p_C, obs=c)

        p_A = D.Normal(means[c], std_devs[c])
        a = numpyro.sample('a', p_A, obs=a)


def generate_IHH_CTR_data_discrete_continuous():
    N = 5000
    with H.seed(rng_seed=0):
        exec_trace = H.trace(model_discete_continuous_IHH_CTR).get_trace(N)

    df = pd.DataFrame({
        'Condition': jax_int_array_to_str_list(
            exec_trace['c']['value'],
            IDX_TO_UNDERLYING_CONDITION,
        ),
        'Telekinetic-Ability': exec_trace['a']['value'],
    })

    df.index.name = 'Patient ID'
    df.to_csv(os.path.join(DATA_DIR, 'IHH-CTR.csv'))


#########################################################################
# IHH Center for Glow and Life Flow: Regression Data
#########################################################################


ABILITY_COEFFS = jnp.array([
    -1.0 / 3840000000.0,
    1.0 / 12000000.0,
    -97.0 / 9600000.0,
    139.0 / 240000.0,
    -19.0 / 1200.0,
    1.0 / 6.0,
    -0.1,
])
ABILITY_STD_DEV = jnp.array(0.03)

GLOW_COEFFS = jnp.array([1.0, -0.01])
GLOW_STD_DEV = jnp.array(0.05)


def model_IHH_CGLF_parameters():
    glow_coefficients = numpyro.param(
        'glow_coefficients',
        GLOW_COEFFS,
        constraint=C.real,
    )

    glow_std_dev = numpyro.param(
        'glow_std_dev',
        GLOW_STD_DEV,
        constraint=C.positive,
    )

    ability_coefficients = numpyro.param(
        'ability_coefficients',
        ABILITY_COEFFS,
        constraint=C.real,        
    )

    ability_std_dev = numpyro.param(
        'ability_std_dev',
        ABILITY_STD_DEV,
        constraint=C.positive,
    )

    return argparse.Namespace(
        glow_coefficients=glow_coefficients,
        glow_std_dev=glow_std_dev,
        ability_coefficients=ability_coefficients,
        ability_std_dev=ability_std_dev,
    )


def model_regression_IHH_CGLF(N, age, glow=None, ability=None):
    parameters = model_IHH_CGLF_parameters()
    
    with numpyro.plate('data', N):        
        p_glow = D.Normal(
            parameters.glow_coefficients[0] + age * parameters.glow_coefficients[1],
            parameters.glow_std_dev,
        )
        glow = numpyro.sample('glow', p_glow, obs=glow)

        p_ability = D.Normal(
            jnp.polyval(parameters.ability_coefficients, age),
            parameters.ability_std_dev,
        )
        ability = numpyro.sample('ability', p_ability, obs=ability)
        

def generate_IHH_CGLF_data_regression():
    N = 500

    age = 20.0 * (1.0 + jax.random.truncated_normal(
        jrandom.PRNGKey(seed=0),
        -1.0,
        4.0,
        shape=(N,),
    ))
    
    with H.seed(rng_seed=1):
        exec_trace = H.trace(model_regression_IHH_CGLF).get_trace(N, age)

    df = pd.DataFrame({
        'Age': age,
        'Glow': exec_trace['glow']['value'],
        'Telekinetic-Ability': exec_trace['ability']['value'],
    })

    df.index.name = 'Patient ID'
    df.to_csv(os.path.join(DATA_DIR, 'IHH-CTR-CGLF-regression-augmented.csv'))

    df.drop(columns=['Age']).to_csv(
        os.path.join(DATA_DIR, 'IHH-CTR-CGLF-regression.csv'),
    )


#########################################################################
# IHH Center for Glow and Life Flow: Regression Data
#########################################################################


def model_classification_IHH_CGLF(
        N,
        age,
        dose,
        glow=None,
        ability=None,
        control_before=None,
        control_after=None,
):
    parameters = model_IHH_CGLF_parameters()
    
    with numpyro.plate('data', N):        
        p_glow = D.Normal(
            parameters.glow_coefficients[0] + age * parameters.glow_coefficients[1],
            parameters.glow_std_dev,
        )
        glow = numpyro.sample('glow', p_glow, obs=glow)

        p_ability = D.Normal(
            jnp.polyval(parameters.ability_coefficients, age),
            parameters.ability_std_dev,
        )
        ability = numpyro.sample('ability', p_ability, obs=ability)

        p_control_before = D.Bernoulli(jnn.sigmoid(30.0 * ability))
        numpyro.sample('control-before', p_control_before, obs=control_before)

        p_control_after = D.Bernoulli(jnn.sigmoid(30.0 * ability + 10.0 * dose))
        numpyro.sample('control-after', p_control_after, obs=control_after)
        

def generate_IHH_CGLF_data_classification():
    key = jrandom.PRNGKey(seed=0)
    key_age, key_dose, key_rest = jrandom.split(key, 3)
    N = 2000

    age = 20.0 * (1.0 + jax.random.truncated_normal(
        key_age,
        -1.0,
        4.0,
        shape=(N,),
    ))

    dose = D.Uniform().sample(key_dose, (N,))
    
    with H.seed(rng_seed=key_rest):
        exec_trace = H.trace(model_classification_IHH_CGLF).get_trace(N, age, dose)

    df = pd.DataFrame({
        'Age': age,
        'Dose': dose,
        'Glow': exec_trace['glow']['value'],
        'Telekinetic-Ability': exec_trace['ability']['value'],
        'Control-Before': exec_trace['control-before']['value'],
        'Control-After': exec_trace['control-after']['value'],
    })

    df.index.name = 'Patient ID'
    df.to_csv(os.path.join(DATA_DIR, 'IHH-CTR-CGLF-classification.csv'))



###############################################################################
# IHH Center for Epidemiology
###############################################################################


def IHH_fever_vs_heart_rate():
    columns = ['Fever', 'Heart-Rate']
    n = 500
    
    z = np.random.randint(0, high=3, size=(n,))
    c1 = np.random.multivariate_normal([-1.0, -1.0], [[0.3, 0.1], [0.1, 0.3]], size=n)
    c2 = np.random.multivariate_normal([1.5, -1.5], [[2.8, -0.2], [-0.2, 3.2]], size=n)
    c3 = np.random.multivariate_normal([0.0, 1.0], [[1., 0.0], [0.0, 0.2]], size=n)

    samples = 0.0
    for idx, c in enumerate([c1, c2, c3]):
        samples += c * np.expand_dims(z == idx, axis=-1)

    df = pd.DataFrame(samples, columns=columns)
    
    df.index.name = 'Patient ID'
    df.to_csv(os.path.join(DATA_DIR, 'IHH-CE-clustering.csv'))

    df = pd.DataFrame(
        sklearn.datasets.make_moons(n_samples=n, noise=0.17)[0],
        columns=columns,
    )

    df.index.name = 'Patient ID'
    df.to_csv(os.path.join(DATA_DIR, 'IHH-sister-CE-clustering.csv'))
    

def IHH_microscope_data():
    '''
    Data from:
    https://huggingface.co/datasets/calcuis/pixel-character
    '''
    images = []
    for fname in sorted(glob.glob('raw/pixel_characters/microscope_img/*.png')):
        im = Image.open(fname).convert('L')
        images.append((jnp.array(im) / 255.0).reshape(1, -1))

    images = jnp.vstack(images)
    jnp.save(os.path.join(DATA_DIR, 'microscope.npy'), images)


    
def main():
    #generate_IHH_ER_data_discrete()
    #generate_IHH_CTR_data_discrete_continuous()
    #generate_IHH_CGLF_data_regression()
    #generate_IHH_CGLF_data_classification()
    #IHH_fever_vs_heart_rate()
    #IHH_microscope_data()
    pass
    
        
if __name__ == '__main__':
    main()

    
