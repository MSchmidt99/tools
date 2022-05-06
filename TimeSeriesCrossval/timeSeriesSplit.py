import numpy as np
import pandas as pd
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

class BetterTSS:
    """ 
    A TimeSeriesSplit which takes in the time column and creates equally sized rolling batches to train on.
    If a time column is passed then the windows will have equal time steps but potentially inequal observation counts.
    """
    
    def __init__(self, time_column=None, train_proportion=0.8, n_folds=5, buffer=0.0, slide=0, bias='right'):
        self.n_folds = n_folds
        self.time_column = time_column
        self.train_proportion = train_proportion
        self.buffer = buffer
        test_prop = 1 - train_proportion
        self.batch_size = (
            1 + (
                test_prop * (slide + 1) * (n_folds - 1)
            )
        )**(-1)
        self.slide = slide
        self.bias = bias

    def split(self, X, y=None, groups=None):
        if self.time_column is None:
            X, y, groups = indexable(X, y, groups)
            n_samples = _num_samples(X)
        else:
            X = self.time_column
            if type(X) is pd.DataFrame:
                X = X[X.keys()[0]]
            X, y, groups = indexable(X, y, groups)
            X_time = np.array(list(dict.fromkeys(X))) # functions the same as set() without jumbling the order
            n_samples = _num_samples(X_time)
        
        if isinstance(self.batch_size, float) and self.batch_size < 1:
            length_per_iter = int(n_samples * self.batch_size)
        elif isinstance(self.batch_size, int) and self.batch_size >= 1:
            length_per_iter = self.batch_size
        else:
            raise ValueError("batch_size must be decimal between 0 and 1.0 or whole number greater than or equal to 1")
        
        test_size = int(length_per_iter * (1 - self.train_proportion))
        buffer_size = int(length_per_iter * self.buffer)
        train_size = length_per_iter - test_size
        
        if self.bias == 'left':
            train_starts = range(0, n_samples - length_per_iter + 1, test_size * (self.slide + 1))
        elif self.bias == 'right':
            overhang = (n_samples - length_per_iter) % (test_size * (self.slide + 1))
            train_starts = range(overhang, n_samples - length_per_iter + 1, test_size * (self.slide + 1))
        else:
            print(f"{self.bias} is not a valid option.")
            raise Exception

        if self.time_column is None:
            indices = np.arange(n_samples)
            for train_start in train_starts:
                yield (
                    indices[train_start : train_start + train_size],
                    indices[train_start + train_size + buffer_size : train_start + length_per_iter],
                )
        else:
            for train_start in train_starts:
                yield (
                    np.concatenate(
                        [
                            np.array([i for i, x2 in enumerate(X) if x == x2])
                            for x in X_time[train_start : train_start + train_size]
                        ]
                    ).astype(int),
                    np.concatenate(
                        [
                            np.array([i for i, x2 in enumerate(X) if x == x2])
                            for x in X_time[train_start + train_size + buffer_size : train_start + length_per_iter]
                        ]
                    ).astype(int),
                )

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_folds