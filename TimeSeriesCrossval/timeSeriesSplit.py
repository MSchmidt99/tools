import numpy as np
import pandas as pd
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

"""
See https://github.com/scikit-learn/scikit-learn/pull/24589 for more up-to-date version.
"""

class RollingWindowCV: # (_BaseKFold)
    """ 
    A variant of TimeSeriesSplit which yields equally sized rolling batches, which
    allows for more consistent parameter tuning.

    If a time column is passed then the windows will be sized according to the time
    steps given without blending (this is useful for longitudinal data).

    Parameters
    ----------
    time_column : Iterable, default=None
        Column of the dataset containing dates. Will function identically with `None`
        when observations are not longitudinal. If observations are longitudinal then
        will facilitate splitting train and validation without date bleeding.

    train_prop : float, default=0.8
        Proportion of each window which should be allocated to training. If
        `buffer_prop` is given then true training proportion will be
        `train_prop - buffer_prop`.
        Validation proportion will always be `1 - train_prop`.

    n_splits : int, default=5
        Number of splits.

    buffer_prop : float, default=0.0
        The proportion of each window which should be allocated to nothing. Cuts into
        `train_prop`.

    slide : float, default=0.0
        `slide + 1` is the number of validation lenghts to step by when generating
        windows. A value between -1.0 and 0.0 will create nearly stationary windows,
        and should be avoided unless for some odd reason it is needed.

    bias : {'left', 'right'}, default='right'
        A 'left' `bias` will yeld indicies beginning at 0 and not necessarily ending
        at N. A 'right' `bias` will yield indicies not necessarily beginning with 0 but
        will however end at N.

    max_long_samples : int, default=None
        If the data is longitudinal and this variable is given, the number of
        observations at each time step will be limited to the first `max_long_samples`
        samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RollingWindowCV
    >>> X = np.random.randn(20, 2)
    >>> y = np.random.randint(0, 2, 20)
    >>> rwcv = RollingWindowCV(n_splits=4)
    >>> for train_index, test_index in rwcv.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
    TRAIN: [ 2  3  4  5  6  7  8  9 10 11] TEST: [12 13]
    TRAIN: [ 4  5  6  7  8  9 10 11 12 13] TEST: [14 15]
    TRAIN: [ 6  7  8  9 10 11 12 13 14 15] TEST: [16 17]
    TRAIN: [ 8  9 10 11 12 13 14 15 16 17] TEST: [18 19]
    >>> # Use a time column with longitudinal data and reduce train proportion
    >>> time_col = np.tile(np.arange(16), 2)
    >>> X = np.arange(64).reshape(32, 2)
    >>> y = np.arange(32)
    >>> rwcv = RollingWindowCV(
    ...     time_column=time_col, train_prop=0.6, n_splits=4, bias='right'
    ... )
    >>> for train_index, test_index in rwcv.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [ 1 17  2 18  3 19  4 20  5 21] TEST: [ 6 22  7 23]
    TRAIN: [ 3 19  4 20  5 21  6 22  7 23] TEST: [ 8 24  9 25]
    TRAIN: [ 5 21  6 22  7 23  8 24  9 25] TEST: [10 26 11 27]
    TRAIN: [ 7 23  8 24  9 25 10 26 11 27] TEST: [12 28 13 29]
    TRAIN: [ 9 25 10 26 11 27 12 28 13 29] TEST: [14 30 15 31]
    >>> # Bias the indicies to the start of the time column
    >>> rwcv = RollingWindowCV(
    ...     time_column=time_col, train_prop=0.6, n_splits=4, bias='left'
    ... )
    >>> for train_index, test_index in rwcv.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [ 0 16  1 17  2 18  3 19  4 20] TEST: [ 5 21  6 22]
    TRAIN: [ 2 18  3 19  4 20  5 21  6 22] TEST: [ 7 23  8 24]
    TRAIN: [ 4 20  5 21  6 22  7 23  8 24] TEST: [ 9 25 10 26]
    TRAIN: [ 6 22  7 23  8 24  9 25 10 26] TEST: [11 27 12 28]
    TRAIN: [ 8 24  9 25 10 26 11 27 12 28] TEST: [13 29 14 30]
    >>> # Introduce a buffer zone between train and validation, and slide window
    >>> # by an additional validation size between windows.
    >>> X = np.arange(25)
    >>> Y = np.arange(25)[::-1]
    >>> rwcv = RollingWindowCV(train_prop=0.6, n_splits=2, buffer_prop=0.2, slide=1.0)
    >>> for train_index, test_index in rwcv.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [2 3 4 5 6 7] TEST: [10 11 12 13 14]
    TRAIN: [12 13 14 15 16 17] TEST: [20 21 22 23 24]
    """
    
    def __init__(self, time_column=None, train_prop=0.8, n_splits=5, buffer_prop=0.0, slide=0.0, bias='right', max_long_samples=None):
        if buffer_prop > train_prop:
            raise ValueError(
                "Buffer proportion cannot be greater than training proportion."
            )
        if slide < -1.0:
            raise ValueError("slide cannot be less than -1.0")

        self.n_splits = n_splits
        self.time_column = time_column
        self.train_prop = train_prop
        self.buffer_prop = buffer_prop
        test_prop = 1 - train_prop
        self.batch_size = (
            1 + (
                test_prop * (slide + 1) * (n_splits - 1)
            )
        )**(-1)
        self.slide = slide
        self.bias = bias
        if max_long_samples is not None:
            max_long_samples += 1 # index slice end is exclusivve
        self.max_long_samples = max_long_samples

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
    
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
    
        Yields
        ------
        train : ndarray
            The training set indices for that split.
    
        test : ndarray
            The testing set indices for that split.
        """
        if self.time_column is None:
            X, y, groups = indexable(X, y, groups)
            n_samples = _num_samples(X)
        else:
            X = self.time_column
            if type(X) is pd.DataFrame:
                X = X[X.keys()[0]]
            X, y, groups = indexable(X, y, groups)
            X_time = np.array(list(dict.fromkeys(X)))
            n_samples = _num_samples(X_time)
        
        if isinstance(self.batch_size, float) and self.batch_size < 1:
            length_per_iter = int(n_samples * self.batch_size)
        elif isinstance(self.batch_size, int) and self.batch_size >= 1:
            length_per_iter = self.batch_size
        else:
            raise ValueError("batch_size must be decimal between 0 and 1.0 or whole"
                             + " number greater than or equal to 1 "
                             +f"(got {self.batch_size}).")
        
        test_size = int(length_per_iter * (1 - self.train_prop))
        buffer_size = int(length_per_iter * self.buffer_prop)
        train_size = length_per_iter - test_size - buffer_size
        
        if self.bias == 'left':
            train_starts = range(
                0, n_samples - length_per_iter + 1,
                int(test_size * (self.slide + 1))
            )
        elif self.bias == 'right':
            overhang = (n_samples - length_per_iter) % int(test_size * (self.slide + 1))
            train_starts = range(
                overhang, n_samples - length_per_iter + 1,
                int(test_size * (self.slide + 1))
            )
        else:
            raise ValueError(f"{self.bias} is not a valid option for bias.")

        if self.time_column is None:
            indices = np.arange(n_samples)
            for train_start in train_starts:
                yield (
                    indices[train_start : train_start + train_size],
                    indices[train_start + train_size + buffer_size :
                            train_start + length_per_iter],
                )
        else:
            for train_start in train_starts:
                yield (
                    np.concatenate(
                        [
                            np.array(
                                [i for i, x2 in enumerate(X) if x == x2]
                            )[:self.max_long_samples]
                            for x in X_time[train_start : train_start + train_size]
                        ]
                    ).astype(int),
                    np.concatenate(
                        [
                            np.array(
                                [i for i, x2 in enumerate(X) if x == x2]
                            )[:self.max_long_samples]
                            for x in X_time[train_start + train_size + buffer_size :
                                            train_start + length_per_iter]
                        ]
                    ).astype(int),
                )

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits
