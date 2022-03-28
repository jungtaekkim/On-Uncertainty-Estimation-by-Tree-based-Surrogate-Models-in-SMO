import numpy as np
import copy

import sklearn.ensemble as skle
from bayeso.gp import gp

import tree_based_surrogates as tbs


def get_model(str_surrogate, X_train, Y_train):
    num_estimators = 100
    rate_oversampling = 4.0
    num_duplicates = int(rate_oversampling * 4.0)
    seed = 42

    X_train_ = copy.deepcopy(X_train)
    Y_train_ = np.squeeze(copy.deepcopy(Y_train), axis=1)

    if str_surrogate == 'ours':
        X_train_ = np.tile(X_train_, (num_duplicates, 1))
        Y_train_ = np.tile(Y_train_, (num_duplicates, ))

        model = skle.ExtraTreesRegressor(
            n_estimators=num_estimators,
            max_features='sqrt',
            bootstrap=True,
            random_state=seed,
            max_samples=rate_oversampling / num_duplicates,
        )

        model.fit(X_train_, Y_train_)

    return model

def get_preds(str_surrogate, X_train, Y_train, X_test):
    if str_surrogate == 'gaussian_process':
        mean, std, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test, str_cov='matern52', fix_noise=False, debug=True)

        mean = np.squeeze(mean, axis=1)
        std = np.squeeze(std, axis=1)
    else:
        model = get_model(str_surrogate, X_train, Y_train)

        mean = model.predict(X_test)
        std = tbs.return_std(X_test, model.estimators_, mean)

    return mean, std
