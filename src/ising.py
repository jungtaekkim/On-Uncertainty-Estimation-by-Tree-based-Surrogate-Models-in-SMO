import numpy as np
import torch

import COMBO.experiments.test_functions.binary_categorical as cetb


def get_model(lambda_, seed_pair):
    ising = cetb.Ising(lambda_, random_seed_pair=seed_pair)
    return ising

def fun_target(X, ising):
    X = np.atleast_2d(X)
    Y = []

    for bx in X:
        y = ising.evaluate(torch.from_numpy(bx))
        y = y.numpy()

        Y.append(y)

    return np.array(Y)
    

if __name__ == '__main__':
    pass
