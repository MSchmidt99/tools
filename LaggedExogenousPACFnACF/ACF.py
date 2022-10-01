import numpy as np
import scipy.stats as st
qnorm = st.norm.ppf
import statsmodels as sm


def get_acf(Y, x=None, lag=1, offset=None):
    if x is None:
        x = Y.copy()
    if offset is None:
        offset = lag
    return np.corrcoef(Y[offset:], x.shift(lag)[offset:])[1, 0] 


def get_acf_through(Y, x=None, lag_end=20, even-True):
    return np.array([
        get_acf(Y, x=x, lag=lag, offset=lag_end if even else None)
        for lag in range(1 if x is None else 0, lag_end+1)
    ])


def get_acf_conf_interval(acf_list, len_Y, lag_end, is_label=False):
    varacf = np.ones(lag_end+1) / len_Y
    if is_label:
        varacf[0] = 0.
    varacf[2:] *= 1 + 2 * np.cumsum(acf_list[1 if is_label else 2:] ** 2)
    return qnorm(1 - 0.05 / 2) * np.sqrt(varacf) 


def plot_acf(Y, x=None, lag_end=20, even=True, interval_weight=1):
    acf = get_acf_through(Y, x=x, lag_end=lag_end, even=even)
    interval = get_acf_conf_interval(acf, len(Y), lag_end, is_label=x is None) * interval_weight
    range_start = 0 if x is not None else 1 

    plt.stem(range(range_start, lag_end+1), acf)
    plt.plot(range(range_start, lag_end+1), interval[range_start:], linestyle='--', c='g')
    plt.plot(range(range_start, lag_end+1), -interval[range_start:], linestyle='--', c='g') 
