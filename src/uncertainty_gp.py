import numpy as np

from bayeso.gp import gp

import utils


parser, args = utils.get_parser()

str_fun = args.function
print(str_fun)

if str_fun == 'few':
    import fun_1d_1 as unc
    str_exp = 'unc_1d_few_gp'
elif str_fun == 'many':
    import fun_1d_2 as unc
    str_exp = 'unc_1d_many_gp'
elif str_fun == 'cubic':
    import fun_1d_3 as unc
    str_exp = 'unc_1d_cubic_gp'
else:
    raise ValueError('not allowed str_fun')

print(str_exp)


if __name__ == '__main__':
    mean, std, Sigma = gp.predict_with_optimized_hyps(unc.X_train, unc.Y_train[..., np.newaxis], unc.X_test, str_cov='matern52', fix_noise=False, debug=True, str_optimizer_method='Nelder-Mead')

    mean = np.squeeze(mean, axis=1)
    std = np.squeeze(std, axis=1)

    mean_gp, std_gp, _ = gp.predict_with_optimized_hyps(unc.X_train, unc.Y_train[..., np.newaxis], unc.X_test, str_cov='matern52', fix_noise=False, debug=True, str_optimizer_method='Nelder-Mead')

    mean_gp = np.squeeze(mean_gp, axis=1)
    std_gp = np.squeeze(std_gp, axis=1)

    nll = utils.compute_nll(mean, std, np.squeeze(unc.X_test, axis=1), unc.Y_test, np.squeeze(unc.X_train, axis=1))
    print('nll {:.4f}'.format(nll))

    kl = utils.compute_kl(mean, std, mean_gp, std_gp)
    print('kl {:.4f}'.format(kl))

    utils.plot_1d(unc.X_train, unc.Y_train, unc.X_test, unc.Y_test, mean, std, str_exp)
