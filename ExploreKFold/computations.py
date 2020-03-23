import numpy as np
import pandas as pd


def data_generator(n, noise='normal', scale=.5, underlying_relation='linear'):
    x = np.linspace(start=0, stop=10, num=n)
    if underlying_relation == 'linear':
        y = x * 1

    elif underlying_relation == 'sine_wave':
        y = 2 * np.sin(x) + 5

    if noise == 'normal':
        y += np.random.normal(size=len(y), scale=scale)
    elif noise == 'constant':
        y += scale

    return pd.DataFrame(dict(x=x, y=y))
