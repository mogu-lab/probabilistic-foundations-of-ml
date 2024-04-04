import pandas as pd
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
import numpyro.distributions as D


IDX_TO_CONDITION = [
    'Broken Limb',    
    'High Fever',
    'Entangled Antennas',
    'Allergic Reaction',
]

IDX_TO_BOOL = [
    'No',
    'Yes',
]


def generate_data_for_q1():
    key = jrandom.PRNGKey(seed=0)
    key_weekday, key_weekend, key_condition, key_hospitalized = jrandom.split(key, 4)
    
    NUM_DAYS = 100

    day = jnp.arange(NUM_DAYS)
    
    p_visits_weekday = D.Poisson(rate=50.0)
    p_visits_weekend = D.Poisson(rate=20.0)

    num_visits = jnp.where(
        (day % 7 == 5) | (day % 7 == 6),
        p_visits_weekend.sample(key_weekend, (NUM_DAYS,)),
        p_visits_weekday.sample(key_weekday, (NUM_DAYS,)),
    )

    p_condition = D.Categorical(jnp.array([0.2, 0.4, 0.1, 0.3]))
    p_hospitalized = D.Bernoulli(jnp.array([0.8, 0.1, 0.1, 0.7]))
    
    conditions = p_condition.sample(key_condition, (int(num_visits.sum()),))
    hospitalized = p_hospitalized.sample(key_condition, (int(num_visits.sum()),))    
    hospitalized = hospitalized[(jnp.arange(conditions.size), conditions)]

    rows = []
    idx = 0
    for day, num_visits in zip(day, num_visits):
        for _ in range(num_visits):
            rows.append({
                'Day': day,
                'Condition': IDX_TO_CONDITION[conditions[idx]],
                'Hospitalized': IDX_TO_BOOL[hospitalized[idx]],
            })

            idx += 1

    df = pd.DataFrame(rows)

    return df

        
def main():
    df = generate_data_for_q1()
    df.to_csv('IHH-ER.csv', index=False)


if __name__ == '__main__':
    main()

    
