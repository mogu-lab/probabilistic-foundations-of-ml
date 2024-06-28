from collections import defaultdict

import pandas as pd
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
import numpyro.distributions as D
import numpyro.distributions.constraints as C
import numpyro.handlers as H
import chex


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
            'Attempts-to-Disentangle': None,
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

    rho_attempts_to_disentangle = numpyro.param(
        'rho_attempts_to_disentangle',
        jnp.array([jnp.nan, 0.3]),
        constraint=C.unit_interval,        
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
        p_attempts_to_disentangle = D.Geometric(
            rho_attempts_to_disentangle[entangled],
        )
        attempts = numpyro.sample(
            'Attempts-to-Disentangle',
            p_attempts_to_disentangle,
            obs=data['Attempts-to-Disentangle']
        )
        chex.assert_shape(attempts, (N,))


def generate_ihh_er_data_discrete():
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
        'Attempts-to-Disentangle': (
            exec_trace['Attempts-to-Disentangle']['value']
        ),
    })

    df.index.name = 'Patient ID'
    df.to_csv('data/IHH-ER.csv')


#########################################################################
# IHH Telekinesis Center Discrete-Continuous Data
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


def generate_ihh_ctr_data_discrete_continuous():
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
    df.to_csv('data/IHH-CTR.csv')


def main():
    generate_ihh_er_data_discrete()
    generate_ihh_ctr_data_discrete_continuous()
    
        
if __name__ == '__main__':
    main()

    
