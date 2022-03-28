import numpy as np


'''
def get_mean_std(model, X_test):
    preds = model.predict(X_test)

    list_preds = []

    for est in model.estimators_:
        list_preds.append(est.predict(X_test))

    mean = np.mean(list_preds, axis=0)
    std = np.std(list_preds, axis=0)
    print(np.linalg.norm(np.abs(preds - mean)))

    return mean, std
'''

def return_std(X, trees, predictions, min_variance=0.0):
    """
    Returns `std(Y | X)`.
    Can be calculated by E[Var(Y | Tree)] + Var(E[Y | Tree]) where
    P(Tree) is `1 / len(trees)`.
    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Input data.
    * `trees` [list, shape=(n_estimators,)]:
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or ExtraTreesRegressor.
    * `predictions` [array-like, shape=(n_samples,)]:
        Prediction of each data point as returned by RandomForestRegressor
        or ExtraTreesRegressor.
    Returns
    -------
    * `std` [array-like, shape=(n_samples,)]:
        Standard deviation of `y` at `X`. If criterion
        is set to "mse", then `std[i] ~= std(y | X[i])`.
    """
    # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
    std = np.zeros(len(X))

    for tree in trees:
        var_tree = tree.tree_.impurity[tree.apply(X)]

        # This rounding off is done in accordance with the
        # adjustment done in section 4.3.3
        # of http://arxiv.org/pdf/1211.0906v2.pdf to account
        # for cases such as leaves with 1 sample in which there
        # is zero variance.
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        std += var_tree + mean_tree ** 2

    std /= len(trees)
    std -= predictions ** 2.0
    std[std < 0.0] = 0.0
    std = std ** 0.5
    return std
