from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

class optimizeableSklearnCVWrapper:
    def __init__(self, model, bounds, cv, error_metric="accuracy", penalize_instability=False, additional_params=None, n_jobs=1, verbose=0):
        """
        model: the input model for which the parameter bounds correspond to.
        bounds: dict of { 'parameter' : ( lower, upper, dtype (optional) ) } (datatype must be a function which can be applied).
                      datatype does not have to be a datatype, functions such as math.exp or a category selection helper 
                      will also work so long as it only has one input parameter and a transformed output. Bayes_opt will
                      sample within the range enclosed between the first two parameters (exclusive of the upper bound,
                      inclusive of the lower bound).
        cv: the cross-validation value or function to use (must be pre-tuned if function).

        Note: cross_validate will return nan if the model errors, which will typically correspond to a parameter input
                      out of bounds error.
        """
        bounds_dtypes = {}
        for key, value in bounds.items():
            if len(value) == 3: # (1, 5, int) = int( (1, 5) ) = (1, 2, 3, 4)
                bounds_dtypes[key] = value[2]      # int()
                bounds[key] = (value[0], value[1]) # (1, 5)
        
        self.model              = model
        self.bounds             = bounds
        self.cv                 = cv
        self.error_metric       = error_metric
        self.additional_params  = additional_params
        self.n_jobs             = n_jobs
        self.verbose            = verbose
        self.bounds_dtypes      = bounds_dtypes
        self.best_score         = 0

    # Helper which returns the model_crossval input parameters as seen by tuned_model
    # (fancy way to say it applies the dtype functions passed in the bounds dict to the values input)
    def format_kwargs(self, kwargs): # applies bounds_dtypes functions to bounds
        for key, value in kwargs.items():
            if key in self.bounds_dtypes:
                kwargs[key] = self.bounds_dtypes.get(key)(value) # int( (1, 5) ) = (1, 2, 3, 4)
        return kwargs
    
    # Returns the cross_val_score for the input parameters when fit on the model passed in __init__() using
    # the data passed in fit().
    def model_crossval(self, return_cv=False, **kwargs):
        start_time = timer()
        kwargs = self.format_kwargs(kwargs)
        if self.verbose >= 3:
            print(kwargs)
        
        if self.additional_params != None:
            kwargs.update(self.additional_params)

        tuned_model = self.model(
            **kwargs, # unpack bounds passed by BayesianOptimization into model 
            # (**{'max_depth' : 20} -> max_depth=20). Will error if param passed does not exist.
            n_jobs=self.n_jobs
        )

        model_cv = cross_validate(
            tuned_model,
            self.X,
            self.y,
            scoring=self.error_metric, 
            cv=self.cv,
            n_jobs=1, # self.n_jobs used in model rather than cv to reduce memory allocation.
            return_train_score=True,
            return_estimator=True
        )
            
        if self.verbose >= 2:
            print("Time elapsed: {0}".format(timedelta(seconds=timer() - start_time)))
        
        mean_cv_score = np.mean(model_cv['test_score'])
        if penalize_instability:
            mean_cv_score -= np.std(model_cv['test_score'])
        if mean_cv_score > self.best_score:
            self.best_score = mean_cv_score
            self.best_model_cv = model_cv
            self.best_params = kwargs

        if return_cv:
            return model_cv
        else:
            if mean_cv_score == np.nan:
                mean_cv_score = 0
            return mean_cv_score
    
    # Creates a BayesianOptimization instance which will optimize model_crossval over the bounds
    # passed in __init__().
    # x and y: the data for the model to fit on.
    # init_points: the number of points randomly sampled within the bounds for the optimizer to start with.
    # n_iter: the number of optimization iterations to complete before selecting the best parameters and stopping
    #         the optimization
    def fit(self, X, y, init_points=1, n_iter=25):
        self.X = X
        self.y = y
        
        self.optimizer = BayesianOptimization(
            f=self.model_crossval,
            pbounds=self.bounds,
            verbose=self.verbose
        )

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
    def get_optimizer(self):
        return self.optimizer

class categorySelector:
    # a simple function which preloads a list of string values to select
    # from iteratively when using a random float as input
    def __init__(self, category_list):
        self.category_list = category_list
        self.max_index = len(self.category_list) - 1
        
    def select(self, value):
        value = int(value)
        if value >= self.max_index + 1:
            value = self.max_index
        if value < 0:
            value = 0
        return self.category_list[value]
