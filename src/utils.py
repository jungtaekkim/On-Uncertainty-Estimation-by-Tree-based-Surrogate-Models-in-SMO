import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_1d(X_train, Y_train, X_test, Y_test, mean=None, std=None, str_figure=None, show_fig=False):
    plt.rc('text', usetex=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ax.plot(X_test, Y_test, linewidth=4)

    if mean is not None:
        line, = ax.plot(X_test, mean, linewidth=4)

    if mean is not None and std is not None:
        ax.fill_between(X_test.flatten(), mean - 1.96 * std, mean + 1.96 * std, alpha=0.25, color=line.get_color())

    ax.plot(X_train, Y_train, 'x', linestyle='none', markersize=10, mew=4)

    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.tick_params(labelsize=24)
    ax.set_xlim([np.min(X_test), np.max(X_test)])
    ax.grid()

    plt.tight_layout()

    if str_figure is not None:
        plt.savefig(
            os.path.join('../figures', str_figure + '.pdf'),
            format='pdf',
            transparent=True
        )

    if show_fig:
        plt.show()

    plt.close('all')

def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--function', type=str)

    args = parser.parse_args()

    return parser, args

def compute_nll(preds_mu, preds_sigma, X_test, Y_test, X_train):
    assert len(preds_mu.shape) == len(preds_sigma.shape) == len(X_test.shape) == len(Y_test.shape) == len(X_train.shape) == 1
    assert preds_mu.shape[0] == preds_sigma.shape[0] == X_test.shape[0] == Y_test.shape[0]

    nll = 0.0

    for mu, sigma, x, y in zip(preds_mu, preds_sigma, X_test, Y_test):
        if np.any(np.abs(X_train - x) < 0.025):
            continue
        log_pdf = norm.logpdf(y, loc=mu, scale=sigma)
        nll -= log_pdf

    nll /= preds_mu.shape[0]

    return nll

def compute_kl(preds_mu, preds_sigma, mean_gp, std_gp):
    assert len(preds_mu.shape) == len(preds_sigma.shape) == len(mean_gp.shape) == len(std_gp.shape) == 1
    assert preds_mu.shape[0] == preds_sigma.shape[0] == mean_gp.shape[0] == std_gp.shape[0]

    kl = 0.0

    for mu, sigma, mu_gp, sigma_gp in zip(preds_mu, preds_sigma, mean_gp, std_gp):
        cur_kl = np.log(sigma_gp / (sigma + 1e-7)) + (sigma**2 + (mu - mu_gp)**2) / (2 * sigma_gp**2) - 1 / 2

        kl = cur_kl

    kl /= preds_mu.shape[0]

    return kl


if __name__ == '__main__':
    pass
