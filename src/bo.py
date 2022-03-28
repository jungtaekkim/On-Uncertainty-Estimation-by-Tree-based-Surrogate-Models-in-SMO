import time
import numpy as np
from scipy.optimize import minimize
import qmcpy

from bayeso import covariance
from bayeso import constants
from bayeso.gp import gp
from bayeso.gp import gp_kernel
from bayeso.utils import utils_bo

import surrogates


def get_best_acquisition_by_evaluation(initials, fun_objective):
    assert isinstance(initials, np.ndarray)
    assert callable(fun_objective)
    assert len(initials.shape) == 2

    acq_vals = fun_objective(initials)

    initial_best = initials[np.argmin(acq_vals, axis=0)]
    initial_best = initial_best[np.newaxis, ...]

    return initial_best


class BO:
    def __init__(self, range_X, str_acq, str_surrogate):
        assert isinstance(range_X, np.ndarray)
        assert isinstance(str_acq, str)
        assert len(range_X.shape) == 2
        assert range_X.shape[1] == 2
        assert (range_X[:, 0] <= range_X[:, 1]).all()
        assert str_acq in constants.ALLOWED_BO_ACQ

        self.range_X = range_X
        self.num_dim = range_X.shape[0]
        self.str_acq = str_acq

        assert str_surrogate in ['gaussian_process', 'ours']
        self.str_surrogate = str_surrogate

    def _get_samples_uniform(self, num_samples, seed=None):
        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))

        if seed is not None:
            state_random = np.random.RandomState(seed)
        else:
            state_random = np.random.RandomState()

        list_initials = []
        for _ in range(0, num_samples):
            list_initial = []
            for elem in self.range_X:
                list_initial.append(state_random.uniform(elem[0], elem[1]))
            list_initials.append(np.array(list_initial))
        initials = np.array(list_initials)
        return initials

    def _get_samples_sobol(self, num_samples, seed=None):
        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))

        sampler = qmcpy.Sobol(self.num_dim, seed=seed, graycode=True)
        samples = sampler.gen_samples(num_samples)

        samples = samples * (self.range_X[:, 1].flatten() - self.range_X[:, 0].flatten()) + self.range_X[:, 0].flatten()
        return samples

    def get_samples(self, str_sampling_method, num_samples=constants.NUM_SAMPLES_AO, seed=None):
        assert isinstance(str_sampling_method, str)
        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))

        if str_sampling_method == 'uniform':
            samples = self._get_samples_uniform(num_samples, seed=seed)
        elif str_sampling_method == 'sobol':
            samples = self._get_samples_sobol(num_samples, seed=seed)
        else:
            raise ValueError('Invalid str_sampling_method.')

        return samples

    def get_initials(self, str_initial_method, num_initials, seed=None):
        assert isinstance(str_initial_method, str)
        assert isinstance(num_initials, int)
        assert isinstance(seed, (int, type(None)))

        return self.get_samples(str_initial_method, num_samples=num_initials, seed=seed)

    def _optimize_objective_gp(self, fun_acquisition, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps):
        X_test = np.atleast_2d(X_test)
        pred_mean, pred_std, _ = gp.predict_with_cov(X_train, Y_train, X_test,
            cov_X_X, inv_cov_X_X, hyps, str_cov='matern52',
            prior_mu=None, debug=True)

        acquisitions = fun_acquisition(pred_mean=np.ravel(pred_mean),
            pred_std=np.ravel(pred_std), Y_train=Y_train)
        return acquisitions

    def _optimize_objective_tree(self, fun_acquisition, X_train, Y_train, X_test, model):
        X_test = np.atleast_2d(X_test)

        pred_mean, pred_std = surrogates.get_preds(self.str_surrogate, X_train, Y_train, X_test)

        acquisitions = fun_acquisition(pred_mean=np.ravel(pred_mean),
            pred_std=np.ravel(pred_std), Y_train=Y_train)
        return acquisitions

    def _get_bounds(self):
        list_bounds = []
        for elem in self.range_X:
            list_bounds.append(tuple(elem))
        return list_bounds

    def _optimize(self, fun_negative_acquisition, str_sampling_method, num_samples):
        list_next_point = []

        if self.str_surrogate == 'gaussian_process':
            list_bounds = self._get_bounds()
            initials = self.get_samples(str_sampling_method, num_samples=num_samples)

            for arr_initial in initials:
                next_point = minimize(
                    fun_negative_acquisition,
                    x0=arr_initial,
                    bounds=list_bounds,
                    method='L-BFGS-B',
                    options={'disp': False}
                )
                next_point_x = next_point.x
                list_next_point.append(next_point_x)
        else:
            list_next_point = self.get_samples(str_sampling_method,
                num_samples=50000)

        next_points = np.array(list_next_point)
        next_point = get_best_acquisition_by_evaluation(
            next_points, fun_negative_acquisition)[0]
        return next_point, next_points

    def optimize(self, X_train, Y_train,
        str_sampling_method='sobol',
        num_samples=100,
        hyps=None
    ):
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(str_sampling_method, str)
        assert isinstance(num_samples, int)
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert num_samples > 0

        time_start = time.time()

        if np.max(Y_train) != np.min(Y_train):
            Y_train = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) * constants.MULTIPLIER_RESPONSE

        time_start_surrogate = time.time()

        if self.str_surrogate == 'gaussian_process':
            if hyps is None:
                cov_X_X, inv_cov_X_X, hyps = gp_kernel.get_optimized_kernel(
                    X_train, Y_train,
                    None, 'matern52',
                    str_optimizer_method='BFGS',
                    str_modelselection_method='ml',
                    use_ard=True,
                    debug=True
                )
            else:
                cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train, hyps, 'matern52', use_gradient=False, debug=True)
        else:
            model = surrogates.get_model(self.str_surrogate, X_train, Y_train)
            hyps = {'noise': None}

        time_end_surrogate = time.time()

        fun_acquisition = utils_bo.choose_fun_acquisition(self.str_acq, hyps['noise'])

        time_start_acq = time.time()

        if self.str_surrogate == 'gaussian_process':
            fun_negative_acquisition = lambda X_test: -1.0 * constants.MULTIPLIER_ACQ * self._optimize_objective_gp(fun_acquisition, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)
        else:
            fun_negative_acquisition = lambda X_test: -1.0 * constants.MULTIPLIER_ACQ * self._optimize_objective_tree(fun_acquisition, X_train, Y_train, X_test, model)

        next_point, next_points = self._optimize(fun_negative_acquisition, str_sampling_method=str_sampling_method, num_samples=num_samples)

        time_end_acq = time.time()

#        acquisitions = fun_negative_acquisition(next_points)

        time_end = time.time()

        dict_info = {
            'next_points': next_points,
#            'acquisitions': acquisitions,
            'time_overall': time_end - time_start,
            'time_surrogate': time_end_surrogate - time_start_surrogate,
            'time_acq': time_end_acq - time_start_acq,
        }

        return next_point, dict_info
