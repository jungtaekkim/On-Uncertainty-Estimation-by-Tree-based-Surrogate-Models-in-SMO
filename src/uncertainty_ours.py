import numpy as np
import copy
import sklearn.ensemble as skle

from bayeso.gp import gp

import tree_based_surrogates as tbs
import utils


parser, args = utils.get_parser()

str_fun = args.function
print(str_fun)

if str_fun == 'few':
    import fun_1d_1 as unc
    str_exp = 'unc_1d_few_ours'
elif str_fun == 'many':
    import fun_1d_2 as unc
    str_exp = 'unc_1d_many_ours'
elif str_fun == 'cubic':
    import fun_1d_3 as unc
    str_exp = 'unc_1d_cubic_ours'
else:
    raise ValueError('not allowed str_fun')

print(str_exp)

rate_oversampling = 4.0
num_duplicates = int(rate_oversampling * 4.0)


if __name__ == '__main__':
    X_train = copy.deepcopy(unc.X_train)
    Y_train = copy.deepcopy(unc.Y_train)

    X_train = np.tile(X_train, (num_duplicates, 1))
    Y_train = np.tile(Y_train, (num_duplicates, ))

    print(X_train.shape, Y_train.shape)

    model = skle.ExtraTreesRegressor(
        n_estimators=100,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        max_samples=rate_oversampling / num_duplicates,
    )
    print(model)
    model.fit(X_train, Y_train)
    mean = model.predict(unc.X_test)
    std = tbs.return_std(unc.X_test, model.estimators_, mean)
#    mean, std = tbs.get_mean_std(model, unc.X_test)

    mean_gp, std_gp, _ = gp.predict_with_optimized_hyps(unc.X_train, unc.Y_train[..., np.newaxis], unc.X_test, str_cov='matern52', fix_noise=False, debug=True, str_optimizer_method='Nelder-Mead')

    mean_gp = np.squeeze(mean_gp, axis=1)
    std_gp = np.squeeze(std_gp, axis=1)

    nll = utils.compute_nll(mean, std, np.squeeze(unc.X_test, axis=1), unc.Y_test, np.squeeze(unc.X_train, axis=1))
    print('nll {:.4f}'.format(nll))

    kl = utils.compute_kl(mean, std, mean_gp, std_gp)
    print('kl {:.4f}'.format(kl))

    utils.plot_1d(unc.X_train, unc.Y_train, unc.X_test, unc.Y_test, mean, std, str_exp)

    for ind_est, est in enumerate(model.estimators_[:4]):
        mean_ = est.predict(unc.X_test)
        utils.plot_1d(unc.X_train, unc.Y_train, unc.X_test, unc.Y_test, mean=mean_, std=None, str_figure='{}_{:03d}'.format(str_exp, ind_est + 1), show_fig=False)
