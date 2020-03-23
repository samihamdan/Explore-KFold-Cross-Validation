import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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


def KFold_split(df_data, df_data_extra, n_splits, shuffle):
    X = df_data.x.values.reshape(-1, 1)
    df = pd.DataFrame(dict(x=[], y=[], data_split=[], in_train_set=[], y_pred=[]))

    splitter = KFold(n_splits=n_splits, shuffle=shuffle)
    for fold_nbr, (train_idx, test_idx) in enumerate(splitter.split(X)):
        data_split = pd.Series(f'fold:{fold_nbr}', index=df_data.index)
        in_train_set = pd.Series('out_fold', index=df_data.index)
        in_train_set[train_idx] = 'in_fold'
        df_fold = pd.DataFrame(dict(x=df_data.x, y=df_data.y,
                                    data_split=data_split,
                                    in_train_set=in_train_set, y_pred=np.full((len(df_data)), np.nan)
                                    ))

        df_extra = df_data_extra.copy()
        df_extra['data_split'] = pd.Series(f'fold:{fold_nbr}', index=df_extra.index)
        df_extra['in_train_set'] = pd.Series('extra_test', index=df_extra.index)

        df = pd.concat([df, pd.concat([df_fold, df_extra])])
    return df
