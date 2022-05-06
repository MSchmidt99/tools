import numpy as np
from collections import OrderedDict

from xgboost.sklearn import XGBClassifier

from bayesOptWrapper import optimizeableSklearnCVWrapper
from bayesOptWrapper import categorySelector as category

def optimizeModel(train, cv, label, n_jobs=1):
    model_o = get_opt_wrapper(cv, n_jobs=n_jobs)

    model_o.fit(
        train.drop([label], axis=1),
        train[label],
        init_points=5,
        n_iter=15
    )
    
    model_cv = model_o.best_model_cv

    return model_cv["estimator"][np.argmin(model_cv["train_score"] - model_cv["test_score"])], model_o

def get_opt_wrapper(cv, n_jobs=1):
    xgb_params = {
        'objective': 'multi:softmax',
        'eval_metric': ['mlogloss', 'merror'],
        'use_label_encoder': False,
        'tree_method': 'hist',
    }

    depth = np.linspace(3, 32, 15).astype(int)
    estim = np.linspace(200, 1000, 15).astype(int)
    params = OrderedDict([
        ('max_depth',          (0,      len(depth), category(depth).select)),
        ('gamma',              (1e-9,   9                                 )),
        ('reg_alpha',          (1e-5,   1,                                )),
        ('reg_lambda',         (1e-5,   10                                )),
        ('min_child_weight',   (1,      20,         int                   )),
        ('n_estimators',       (0,      len(estim), category(estim).select)),
        ('learning_rate',      (0.001,  0.5                               ))
    ])

    xgb_o = optimizeableSklearnCVWrapper(
        XGBClassifier,
        params,
        cv,
        verbose=2,
        n_jobs=n_jobs,
        additional_params=xgb_params,
        cuda_params=cuda_params,
    )

    return xgb_o
