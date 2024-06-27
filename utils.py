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


