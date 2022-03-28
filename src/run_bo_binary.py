import os
import time
import numpy as np
import argparse

from bayeso import constants
from bayeso.gp import gp_kernel
from bayeso.utils import utils_bo

import bo


parser = argparse.ArgumentParser(description='')
parser.add_argument('-f', '--function', type=str)
parser.add_argument('-s', '--surrogate', type=str)
parser.add_argument('-i', '--iteration', type=int)

args = parser.parse_args()

path_results = '../results'

if not os.path.exists(path_results):
    os.mkdir(path_results)

str_function = args.function
str_surrogate = args.surrogate
seed = 42
num_samples_ao = 10
str_acq = 'ei'

num_bo = 10
num_init = 5
num_iter = int(args.iteration)

print(str_function)
print(str_surrogate)
print(num_bo, num_init, num_iter)

str_file = 'exp_{}_{}_{}_seed_{}_bo_{}_init_{}_iter_{}.npy'.format(str_function, str_acq, str_surrogate, seed, num_bo, num_init, num_iter)
print(str_file)

path_all = os.path.join(path_results, str_file)

if str_function == 'ising-2':
    import ising
    get_model = lambda seed_pair: ising.get_model(1e-2, seed_pair)
    fun_target = ising.fun_target

    ranges = np.array([[0.0, 1.0]] * 24)
    seed_pairs = np.random.RandomState(seed).randint(0, 10000, size=(num_bo, 2))
elif str_function == 'ising-1':
    import ising
    get_model = lambda seed_pair: ising.get_model(1e-1, seed_pair)
    fun_target = ising.fun_target

    ranges = np.array([[0.0, 1.0]] * 24)
    seed_pairs = np.random.RandomState(seed).randint(0, 10000, size=(num_bo, 2))
elif str_function == 'ising0':
    import ising
    get_model = lambda seed_pair: ising.get_model(1e0, seed_pair)
    fun_target = ising.fun_target

    ranges = np.array([[0.0, 1.0]] * 24)
    seed_pairs = np.random.RandomState(seed).randint(0, 10000, size=(num_bo, 2))
elif str_function == 'contamination-2':
    import contamination
    get_model = lambda seed_pair: contamination.get_model(1e-2, seed_pair)
    fun_target = contamination.fun_target

    ranges = np.array([[0.0, 1.0]] * 25)
    seed_pairs = np.random.RandomState(seed).randint(0, 10000, size=(num_bo, 2))
elif str_function == 'contamination-1':
    import contamination
    get_model = lambda seed_pair: contamination.get_model(1e-1, seed_pair)
    fun_target = contamination.fun_target

    ranges = np.array([[0.0, 1.0]] * 25)
    seed_pairs = np.random.RandomState(seed).randint(0, 10000, size=(num_bo, 2))
elif str_function == 'contamination0':
    import contamination
    get_model = lambda seed_pair: contamination.get_model(1e0, seed_pair)
    fun_target = contamination.fun_target

    ranges = np.array([[0.0, 1.0]] * 25)
    seed_pairs = np.random.RandomState(seed).randint(0, 10000, size=(num_bo, 2))
else:
    raise ValueError('Invalid function.')


def round_bx(bx):
    bx_ = bx >= 0.5
    return bx_.astype(np.float32)


def run_single_round_with_all_initial_information(model_bo, fun_target, X_train, Y_train, num_iter, str_sampling_method_ao, num_samples_ao):
    time_start = time.time()

    X_final = X_train
    Y_final = Y_train
    time_all_final = []
    time_surrogate_final = []
    time_acq_final = []
    time_overall_final = []
    for ind_iter in range(0, num_iter):
        print('ITER: {}'.format(ind_iter + 1))

        time_iter_start = time.time()

        if model_bo.str_surrogate == 'gaussian_process':
            time_start_surrogate_ = time.time()

            if ind_iter < 10 or ind_iter % 100 == 99 or ind_iter % 100 == 0 or ind_iter % 100 == 49 or ind_iter % 100 == 50:
                _, _, hyps = gp_kernel.get_optimized_kernel(
                    X_final, Y_final,
                    None, 'matern52',
                    str_optimizer_method='BFGS',
                    str_modelselection_method='ml',
                    use_ard=True,
                    debug=True
                )

            time_end_surrogate_ = time.time()

            next_point, dict_info = model_bo.optimize(X_final, Y_final,
                str_sampling_method=str_sampling_method_ao,
                num_samples=num_samples_ao, hyps=hyps)

#            dict_info['time_surrogate'] += time_end_surrogate_ - time_start_surrogate_
        else:
            next_point, dict_info = model_bo.optimize(X_final, Y_final,
                str_sampling_method=str_sampling_method_ao,
                num_samples=num_samples_ao)

        next_point = round_bx(next_point)

        next_points = dict_info['next_points']
        time_surrogate = dict_info['time_surrogate']
        time_acq = dict_info['time_acq']
        time_overall = dict_info['time_overall']

#        if np.where(np.linalg.norm(next_point - X_final, axis=1) < 1e-4)[0].shape[0] > 0:
#            next_point = utils_bo.get_next_best_acquisition(next_points, acquisitions, X_final)

        X_final = np.vstack((X_final, next_point))

        time_to_evaluate_start = time.time()
        Y_final = np.vstack((Y_final, fun_target(next_point)))
        time_to_evaluate_end = time.time()

        time_iter_end = time.time()
        time_all_final.append(time_iter_end - time_iter_start)
        time_surrogate_final.append(time_surrogate)
        time_acq_final.append(time_acq)
        time_overall_final.append(time_overall)

    time_end = time.time()

    time_all_final = np.array(time_all_final)
    time_surrogate_final = np.array(time_surrogate_final)
    time_acq_final = np.array(time_acq_final)
    return X_final, Y_final, time_all_final, time_surrogate_final, time_acq_final

def run_single_round(model_bo, fun_target, num_init, num_iter, str_initial_method_bo, str_sampling_method_ao, num_samples_ao, seed):
    X_train = model_bo.get_initials(str_initial_method_bo, num_init, seed=seed)
    Y_train = []
    time_initials = []

    for elem in X_train:
        elem = round_bx(elem)
        time_initial_start = time.time()
        Y_train.append(fun_target(elem)[0])
        time_initial_end = time.time()
        time_initials.append(time_initial_end - time_initial_start)

    time_initials = np.array(time_initials)

    Y_train = np.array(Y_train)
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))

    X_final, Y_final, time_all_final, time_surrogate_final, time_acq_final = run_single_round_with_all_initial_information(
            model_bo,
            fun_target,
            X_train,
            Y_train,
            num_iter,
            str_sampling_method_ao,
            num_samples_ao,
        )
    
    time_all_final = np.concatenate((time_initials, time_all_final))

    return X_final, Y_final, time_all_final, time_surrogate_final, time_acq_final


if __name__ == '__main__':
    if not os.path.exists(path_all):
        Xs = []
        Ys = []
        times_all = []
        times_surrogate = []
        times_acq = []

        for ind_bo in range(0, num_bo):
            cur_seed = seed * (ind_bo + 1)
            print('BO: {} SEED: {}'.format(ind_bo + 1, cur_seed))
            model_bo = bo.BO(ranges, str_acq, str_surrogate)
            model_fun = get_model(seed_pairs[ind_bo])
            fun_target_ = lambda X: fun_target(X, model_fun)

            X_, Y_, time_all, time_surrogate, time_acq = run_single_round(model_bo, fun_target_, num_init, num_iter, 'sobol', 'sobol', num_samples_ao, cur_seed)

            Xs.append(X_)
            Ys.append(Y_)
            times_all.append(time_all)
            times_surrogate.append(time_surrogate)
            times_acq.append(time_acq)

            print(np.min(np.squeeze(Y_)))
            print(np.sum(time_all))

        dict_all = {
            'num_bo': num_bo,
            'num_init': num_init,
            'num_iter': num_iter,
            'str_function': str_function,
            'str_surrogate': str_surrogate,
            'str_acq': str_acq,
            'seed': seed,
            'X': np.array(Xs),
            'Y': np.array(Ys),
            'time_all': np.array(times_all),
            'time_surrogate': np.array(times_surrogate),
            'time_acq': np.array(times_acq),
        }

        np.save(path_all, dict_all)
    else:
        print('{} exists.'.format(str_file))
