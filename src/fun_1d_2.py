import numpy as np
import matplotlib.pyplot as plt

import qmcpy

import utils


RANGES = np.array([-5.0, 5.0])

sampler = qmcpy.Sobol(1, seed=42, graycode=True)

X_train = sampler.gen_samples(50)
X_train = (RANGES[1] - RANGES[0]) * X_train + RANGES[0]

X_test = np.linspace(RANGES[0], RANGES[1], 201)[..., np.newaxis]

FUN_TARGET = np.sin

Y_train = FUN_TARGET(X_train) + np.random.RandomState(42).randn(*X_train.shape) * 0.2
Y_test = FUN_TARGET(X_test)

Y_train = np.squeeze(Y_train, axis=1)
Y_test = np.squeeze(Y_test, axis=1)


if __name__ == '__main__':
    print(X_train)
    print(X_test)
    print(Y_train)
    print(Y_test)

    utils.plot_1d(X_train, Y_train, X_test, Y_test)
