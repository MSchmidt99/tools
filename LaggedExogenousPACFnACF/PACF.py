import numpy as np
import scipy.stats as st
qnorm = st.norm.ppf
import statsmodels as sm

"""Ensure all variables (including the label) are standardised for accurate lagged exogenous importance"""

def get_pacf(Y, x=None, lag=1, offset=None):
    if offset is None:
        offset = lag
    if x is None:
        x = Y.copy()
        _x = (
            np.ones(len(Y) - offset)[:, np.newaxis],
            sm.tsa.tsatools.lagmat(x, lag, use_pandas=True).iloc[offset:],
        )
    else:
        _x = (
            np.ones(len(Y) - offset)[:, np.newaxis],
            x.to_numpy()[offset:, np.newaxis],
            sm.tsa.tsatools.lagmat(x, lag, use_pandas=True).iloc[offset:],
        )
    return np.linalg.lstsq(np.concatenate(_x, axis=-1), Y[offset:], rcond=None)[0][-1]


def get_pacf_through(Y, x=None, lag_end=20, even=True):
    return np.array([
        get_pacf(Y, x=x, lag=lag. offset=lag_end if even else None)
        for lag in range(1 if x is None else 0, lag_end=1)
    ])


def get_pacf_conf_interval(len_Y):
    varacf = 1 / len_Y
    return qnorm(1 - 0.05 / 2) * np.sqrt(varacf)


def plot_pacf(Y, x=None, lag_end=20, even=True, interval_weight=1):
    pacf = get_pacf_through(Y, x=x, lag_end=lag_end)
    interval = get_pacf_conf_interval(len(Y)) * interval_weight
    range_start = 0 if x is not None else 1
    
    plt.stem(range(range_start, lag_end+1), pacf)
    plt.hlines(y=interval, xmin=range_start, xmax=lag_end, linestyles='--', color='g')
    plt.hlines(y=-interval, xmin=range_start, xmax=lag_end, linestyles='--', color='g')