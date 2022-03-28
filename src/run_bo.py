import os
import time
import numpy as np
import argparse

from bayeso import constants
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
num_samples_ao = 100
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

if str_function == 'branin':
    from bayeso_benchmarks.two_dim_branin import Branin as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'hartmann6d':
    from bayeso_benchmarks.six_dim_hartmann6d import Hartmann6D as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'hartmann3d':
    from bayeso_benchmarks.three_dim_hartmann3d import Hartmann3D as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'eggholder':
    from bayeso_benchmarks.two_dim_eggholder import Eggholder as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'beale':
    from bayeso_benchmarks.two_dim_beale import Beale as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'sixhumpcamel':
    from bayeso_benchmarks.two_dim_sixhumpcamel import SixHumpCamel as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'threehumpcamel':
    from bayeso_benchmarks.two_dim_threehumpcamel import ThreeHumpCamel as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'bohachevsky':
    from bayeso_benchmarks.two_dim_bohachevsky import Bohachevsky as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'goldsteinprice':
    from bayeso_benchmarks.two_dim_goldsteinprice import GoldsteinPrice as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'ackley':
    from bayeso_benchmarks.inf_dim_ackley import Ackley as target_fun
    obj = target_fun(4)
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'rosenbrock':
    from bayeso_benchmarks.inf_dim_rosenbrock import Rosenbrock as target_fun
    obj = target_fun(4)
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'holdertable':
    from bayeso_benchmarks.two_dim_holdertable import HolderTable as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'gramacyandlee2012':
    from bayeso_benchmarks.one_dim_gramacyandlee2012 import GramacyAndLee2012 as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'michalewicz':
    from bayeso_benchmarks.two_dim_michalewicz import Michalewicz as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'sphere':
    from bayeso_benchmarks.inf_dim_sphere import Sphere as target_fun
    obj = target_fun(4)
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
elif str_function == 'dropwave':
    from bayeso_benchmarks.two_dim_dropwave import DropWave as target_fun
    obj = target_fun()
    fun_target = lambda inputs: obj.output(inputs)
    ranges = obj.get_bounds()
else:
    raise ValueError('Invalid function.')


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

        next_point, dict_info = model_bo.optimize(X_final, Y_final,
            str_sampling_method=str_sampling_method_ao,
            num_samples=num_samples_ao)

        next_points = dict_info['next_points']
#        acquisitions = dict_info['acquisitions']
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
        time_initial_start = time.time()
        Y_train.append(fun_target(elem))
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

            X_, Y_, time_all, time_surrogate, time_acq = run_single_round(model_bo, fun_target, num_init, num_iter, 'sobol', 'sobol', num_samples_ao, cur_seed)

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
