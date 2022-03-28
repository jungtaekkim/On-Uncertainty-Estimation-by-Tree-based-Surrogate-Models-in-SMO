import numpy as np
import matplotlib.pyplot as plt

import utils


RANGES = np.array([-5.0, 5.0])

X_train = np.array([
    -3.0,
    -1.0,
    0.5,
    1.5,
    4.95,
])[..., np.newaxis]

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
