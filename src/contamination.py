import numpy as np
import torch

import COMBO.experiments.test_functions.binary_categorical as cetb


def get_model(lambda_, seed_pair):
    model = cetb.Contamination(lambda_, random_seed_pair=seed_pair)
    return model

def fun_target(X, model):
    X = np.atleast_2d(X)
    Y = []

    for bx in X:
        y = model.evaluate(torch.from_numpy(bx))
        y = y.numpy()

        Y.append(y)

    return np.array(Y)
    

if __name__ == '__main__':
    pass
