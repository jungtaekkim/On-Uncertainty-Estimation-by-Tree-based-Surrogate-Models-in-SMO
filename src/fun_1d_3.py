import numpy as np
import matplotlib.pyplot as plt
import qmcpy

import utils


RANGES = np.array([-10.0, 10.0])
RANGES_ = np.array([-7.5, 7.5])

sampler = qmcpy.Sobol(1, seed=42, graycode=True)
samples = sampler.gen_samples(10)

X_train = samples * (RANGES_[1] - RANGES_[0]) + RANGES_[0]
X_test = np.linspace(RANGES[0], RANGES[1], 201)[..., np.newaxis]

FUN_TARGET = lambda X: X**3

Y_train = FUN_TARGET(X_train) + np.random.RandomState(42).randn(*X_train.shape) * 100.0
Y_test = FUN_TARGET(X_test)

Y_train = np.squeeze(Y_train, axis=1)
Y_test = np.squeeze(Y_test, axis=1)


if __name__ == '__main__':
    print(X_train)
    print(X_test)
    print(Y_train)
    print(Y_test)

    print(X_train.shape)
    print(Y_train.shape)

    utils.plot_1d(X_train, Y_train, X_test, Y_test)
