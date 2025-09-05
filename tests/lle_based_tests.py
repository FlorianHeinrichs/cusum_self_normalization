#
# tests/lrv_based_tests.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-15
# Author: Florian Heinrichs
#
# Tests based on local linear estimation:
# 1. Bücher, A., Dette, H., & Heinrichs, F. (2021). Are deviations in a
#    gradually varying mean relevant? A testing approach based on sup-norm
#    estimators. The Annals of Statistics, 49(6), 3583-3617.
# 2. Heinrichs, F., & Dette, H. (2021). A distribution free test for changes in
#    the trend function of locally stationary processes. Electronic Journal of
#    Statistics, 15(2), 3762-3797.

from time import time

import numpy as np
from scipy import signal

from .utils import bandwidth_cv, estimate_global_lrv, convolve


def kernel_based_tests(X: np.ndarray,
                       alpha: float = 0.05,
                       n_trajectories: int = 1000,
                       return_p_value: bool = False) -> tuple:
    """
    Wrapper function to do all three kernel based tests from [1] and [2].

    :param X: Time series given as NumPy array of shape (n_time,) or
        (n_time, n_ts), where n_time is the length of the time series and n_ts
        the number of observed time series.
    :param alpha: Level of test.
    :param n_trajectories: Number of repetitions/trajectories, used for
        approximation of quantiles.
    :param return_p_value: Indicator if p-value shall be returned instead of
        test decision.
    :return: Tuple of dictionaries with test decisions (True if H_0 is rejected)
        and times.
    """
    n = X.shape[0]

    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=1)

    min_bw, max_bw = 2, int(n // 4)

    trajectories = {}
    for bw in range(min_bw, max_bw + 1):
        trajectories[bw] = np.array([simulate_gaussian(n, bw / n)
                                     for _ in range(n_trajectories)])

    quantile = estimate_quantile_self_norm(0.05, n_repetitions=n_trajectories)

    estimator = lambda *y: jackknife_estimator(*y)[0]
    bandwidths = bandwidth_cv(X, min_bw, max_bw, estimator, batch_axis=1)

    test_decisions = {'Rel_Dev_Gumbel': [],
                      'Rel_Dev_Gauss': [],
                      'L2_Self_Norm': []}
    time_rel_dev = []
    time_l2_sn = []

    for bw, x in zip(bandwidths, X.transpose()):
        traj = trajectories[bw]

        start = time()
        results = calculate_test_rel(x, traj, bw, alpha=alpha,
                                     return_p_value=return_p_value)
        time_rel_dev.append(time() - start)

        test_decisions['Rel_Dev_Gumbel'].append(results[0])
        test_decisions['Rel_Dev_Gauss'].append(results[1])

        start = time()
        result = calculate_test_self_norm(x, bw, quantile,
                                          return_p_value=return_p_value)
        time_l2_sn.append(time() - start)
        test_decisions['L2_Self_Norm'].append(result)

    times = {'Rel_Dev': time_rel_dev, 'L2_Self_Norm': time_l2_sn}

    return test_decisions, times


def simulate_gaussian(n: int, hn: float) -> np.ndarray:
    """
    Auxiliary function for the approximation of the quantiles of G_{n, 1}.

    :param n: Length of time series
    :param hn: Bandwidth of time series
    :return: Simulated trajectory of G_{n, 1}.
    """
    rng = np.random.default_rng()
    V = rng.normal(size=n)

    bw = int(n * hn)
    _, kernel = get_kernel(bw, version='jackknife')

    kernel_norm = np.sqrt(np.sum(kernel ** 2) / bw)
    trajectory = signal.convolve(V, kernel, mode="same")
    trajectory = np.abs(trajectory / (kernel_norm * np.sqrt(bw)))

    return trajectory


def calculate_test_rel(X: np.ndarray,
                       Y: np.ndarray,
                       bw: int,
                       alpha: float = 0.05,
                       return_p_value: bool = False) -> tuple:
    """
    Calculate test statistics of tests from [1].

    :param X: Time series given as NumPy array of size (n,).
    :param Y: Simulated trajectories of Gaussian process as NumPy array.
    :param bw: Bandwidth of kernel estimator.
    :param alpha: Level of test.
    :param return_p_value: Indicator if p-value shall be returned instead of
        test decision.
    :return: Test decision of two tests (based on Gumbel quantile and Gaussian
        process).
    """
    n = len(X)
    hn = bw / n

    mu_tilde = jackknife_estimator(X, bw)[0]
    lrv = estimate_global_lrv(X, mu_tilde)

    mu_tilde = mu_tilde - np.mean(X)

    support, kernel = get_kernel(bw, version='jackknife')
    _, kernel_2nd_drv = get_kernel_2nd_drv(bw, version='jackknife')

    kernel_norm = np.sqrt(np.sum(kernel ** 2) / bw)
    gamma = np.sqrt(-np.sum(kernel * kernel_2nd_drv) / bw) / (kernel_norm * lrv)

    ln = np.sqrt(2 * np.log(1 + gamma / (2 * np.pi * bw / n)))
    dn = ln / np.sqrt(bw)
    mu_sup = np.max(np.abs(mu_tilde[bw:(n - bw)]))
    E_dn = np.intersect1d(np.where(mu_sup - np.abs(mu_tilde) <= dn)[0],
                          np.arange(bw, n - bw + 1))
    lambda_hat = len(E_dn) / n
    ln_hat = np.sqrt(2 * np.log(1 + gamma * lambda_hat
                                / ((1 - 2 * hn) * 2 * np.pi * hn)))

    Z = np.max(Y[:, E_dn], axis=1)
    multiplier = lrv * kernel_norm / np.sqrt(bw)

    if return_p_value:
        ts_gumbel = ln_hat * mu_sup / multiplier - ln_hat ** 2
        p_value_gumbel = 1 - np.exp(-np.exp(-(ts_gumbel - np.log(2))))
        ts_gaussian = mu_sup / multiplier
        p_value_gaussian = np.mean(ts_gaussian < Z)

        return p_value_gumbel, p_value_gaussian
    else:
        L = Y.shape[0]
        q_alpha = -np.log(-np.log(1 - alpha)) + np.log(2)

        test_gumbel = mu_sup > (q_alpha + ln_hat ** 2) * multiplier / ln_hat
        test_gaussian = mu_sup > np.sort(Z)[int((1 - alpha) * L)] * multiplier

        return test_gumbel, test_gaussian


def sequential_lle(X: np.ndarray,
                   bw: int,
                   bn: int = 20,
                   n_lambda: int = 5) -> np.ndarray:
    """
    Calculate sequential local linear estimator, as used in [2] for
    self-normalization.

    :param X: Time series given as numpy array of size (n_time, n_ts).
    :param bw: Bandwidth of the estimator as int.
    :param bn: Block length.
    :param n_lambda: Number of support points of nu. Calculate sequential
        estimator at grid points 1/n_lambda, ..., 1.
    :return: Returns the sequential local linear estimators as numpy array of
        size (n_time, n_lambda, n_ts) or (n_time, n_lambda) if only one time
        series is provided.
    """
    n = X.shape[0]
    ln = int(n // bn)

    if bn % n_lambda > 0:
        raise ValueError(f"{bn=} and {n_lambda=} not compatible.")

    ones = np.ones(bn // n_lambda)
    zeros = np.zeros(bn // n_lambda)
    filter_arrays = np.stack([
        np.tile(
            np.concatenate((np.tile(ones, i), np.tile(zeros, n_lambda - i))), ln
        ) for i in range(1, n_lambda + 1)
    ])

    n_rem = int(n - bn * ln)
    if n_rem > 0:
        final = np.zeros((n_lambda, n_rem))
        final[-1] = 1
        filter_arrays = np.concatenate((filter_arrays, final), axis=1)

    filter_arrays = filter_arrays.astype(bool)
    estimates = np.stack([jackknife_estimator(X, bw, filter_array=filter_array)[0]
                          for filter_array in filter_arrays], axis=1)

    return estimates


def local_linear_estimator(X: np.ndarray,
                           bw: int,
                           filter_array: np.ndarray = None) -> tuple:
    """
    Use local linear regression to estimate mu and its Frechet derivative.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator as int.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the local linear estimators.
    """
    spatial_dims = len(X.shape) - 1
    n_time = X.shape[0]

    if filter_array is None:
        filter_array = np.ones_like(X, dtype=bool)

    X_filtered = X.copy()
    X_filtered[~filter_array] = 0

    X_support = np.ones_like(X_filtered)
    X_support[~filter_array] = 0

    kernel_support, kernel = get_kernel(bw)
    kernel = kernel.reshape((1,) * spatial_dims + (-1,))
    supp_kern = kernel_support * kernel
    supp2_kern = kernel_support ** 2 * kernel

    padding = ((bw, bw),) + spatial_dims * ((0, 0),)
    X_filtered = np.pad(X_filtered, padding, mode='edge')
    X_support = np.pad(X_support, padding, mode='edge')

    S0 = convolve(X_support, kernel)
    S1 = convolve(X_support, supp_kern[..., ::-1])
    S2 = convolve(X_support, supp2_kern[..., ::-1])

    R0 = convolve(X_filtered, kernel)
    R1 = convolve(X_filtered, supp_kern[..., ::-1])

    denominator = S0 * S2 - S1 ** 2
    mu_hat = (S2 * R0 - S1 * R1) / (denominator + 1e-10)
    mu_prime_hat = (S0 * R1 - S1 * R0) / (bw / n_time * denominator + 1e-10)

    return mu_hat, mu_prime_hat


def jackknife_estimator(X: np.ndarray,
                        bw: int,
                        filter_array: np.ndarray = None) -> tuple:
    """
    Function to calculate the Jackknife version of the local linear estimators.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the Jackknife estimators given as numpy array.
    """
    mu_hat, mu_prime_hat = local_linear_estimator(
        X, bw, filter_array=filter_array
    )

    mu_hat2, mu_prime_hat2 = local_linear_estimator(
        X, int(bw / np.sqrt(2)), filter_array=filter_array
    )

    mu_tilde = 2 * mu_hat2 - mu_hat
    mu_prime_tilde = (np.sqrt(2) / (np.sqrt(2) - 1) * mu_prime_hat2
                      - mu_prime_hat / (np.sqrt(2) - 1))

    return mu_tilde, mu_prime_tilde


def get_kernel(bw: int,
               mode: str = 'quartic',
               version: str = 'regular') -> (np.ndarray, np.ndarray):
    """
    Define kernel for kernel based estimators.

    :param bw: Bandwidth of the estimator as int.
    :param mode: Mode of the kernel given as string. Currently only the
        'quartic', 'triweight', and 'tricube' kernels are supported.
    :param version: Version of the kernel, either 'regular' or 'jackknife'.
    :return: Returns the kernel and its support as numpy arrays.
    :raises: ValueError if unsupported mode is chosen.
    """
    if version == 'regular':
        support = np.arange(-bw, bw + 1) / bw

        if mode == 'quartic':
            kernel = 15 / 16 * (1 - support ** 2) ** 2
        elif mode == 'triweight':
            kernel = 35 / 32 * (1 - support ** 2) ** 3
        elif mode == 'tricube':
            kernel = 70 / 81 * (1 - np.abs(support) ** 3) ** 3
        elif mode == 'triangular':
            kernel = (1 - np.abs(support))
        else:
            raise ValueError(f"{mode=} unknown.")

    elif version == 'jackknife':
        bw2 = int(bw // np.sqrt(2))
        support, kern = get_kernel(bw)
        kern2 = np.sqrt(8) * get_kernel(bw2)[1]
        n_diff = (len(kern) - len(kern2)) // 2
        kernel = np.pad(kern2, (n_diff, n_diff)) - kern

    else:
        raise ValueError(f"{version=} unknown.")

    return support, kernel

def get_kernel_2nd_drv(bw: int,
                       mode: str = 'quartic',
                       version: str = 'regular') -> (np.ndarray, np.ndarray):
    """
    Get 2nd derivative of kernel for kernel based estimators.

    :param bw: Bandwidth of the estimator as int.
    :param mode: Mode of the kernel given as string. Currently only the
        'quartic' kernel is supported.
    :param version: Version of the kernel, either 'regular' or 'jackknife'.
    :return: Returns the kernel's 2nd derivative and its support as numpy arrays.
    :raises: ValueError if unsupported mode is chosen.
    """
    if version == 'regular':
        support = np.arange(-bw, bw + 1) / bw

        if mode == 'quartic':
            kernel = 15 / 4 * (3 * support ** 2 - 1)
        else:
            raise ValueError(f"{mode=} unknown.")
    elif version == 'jackknife':
        bw2 = int(bw // np.sqrt(2))
        support, kern = get_kernel_2nd_drv(bw)
        kern2 = np.sqrt(8) * get_kernel_2nd_drv(bw2)[1]
        n_diff = (len(kern) - len(kern2)) // 2
        kernel = np.pad(kern2, (n_diff, n_diff)) - kern

    else:
        raise ValueError(f"{version=} unknown.")


    return support, kernel


def calculate_test_self_norm(X: np.ndarray,
                             bw: int,
                             quantile: float,
                             return_p_value: bool = False) -> bool | float:
    """
    Calculate test statistic of test from [2].

    :param X: Time series given as NumPy array of size (n,).
    :param bw: Bandwidth of kernel estimator.
    :param quantile: Quantile of limiting distribution, obtained from
        estimate_quantile_self_norm().
    :param return_p_value: Indicator if p-value shall be returned instead of
        test decision.
    :return: Test decision of test.
    """
    mu_tilde = sequential_lle(X, bw)
    mean = np.mean(X, axis=0, keepdims=True)
    mu_tilde = mu_tilde - np.expand_dims(mean, axis=-1)

    d2_discrete = np.sqrt(np.mean(mu_tilde ** 2, axis=0))
    lambda_values = np.arange(1, 5) / 5

    integral_term = np.mean(
        lambda_values * np.abs(d2_discrete[:-1] ** 2 - d2_discrete[-1] ** 2),
        axis=0
    )

    test_statistic = d2_discrete[-1] ** 2 / integral_term

    if return_p_value:
        p_value = estimate_p_value_self_norm(test_statistic)
        return p_value
    else:
        reject = test_statistic > quantile
        return reject

def estimate_quantile_self_norm(alpha: float,
                                n_lambda: int = 5,
                                n_grid: int = 1000,
                                n_repetitions: int = 1000) -> float:
    """
    Monte Carlo simulation for quantiles of the test statistic from [2].

    :param alpha: Level of test.
    :param n_lambda: Number of support points of nu. Calculate sequential
        estimator at grid points 1/n_lambda, ..., 1.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param n_repetitions: Number of repetitions of Monte Carlo simulation.
    :return: Approximated quantile.
    """
    rng = np.random.default_rng()

    increments = (rng.standard_normal(size=(n_repetitions, n_grid))
                  / np.sqrt(n_grid))
    increments[:, 0] = 0
    W = np.cumsum(increments, axis=-1)

    lambda_val = np.arange(1, n_lambda) / n_lambda
    lambda_ind = (lambda_val * n_grid - 1).astype(int)
    B = W[:, -1] / np.mean(
        np.abs(W[:, lambda_ind] - lambda_val[None, :] * W[:, -1, None]), axis=-1
    )

    quantile = np.sort(B)[int(n_repetitions * (1 - alpha)) - 1]

    return quantile


def estimate_p_value_self_norm(test_statistic: np.ndarray,
                               n_lambda: int = 5,
                               n_grid: int = 1000,
                               n_repetitions: int = 1000) -> float:
    """
    Monte Carlo simulation for quantiles of the test statistic from [2].

    :param test_statistic: Test statistic as Numpy array.
    :param n_lambda: Number of support points of nu. Calculate sequential
        estimator at grid points 1/n_lambda, ..., 1.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param n_repetitions: Number of repetitions of Monte Carlo simulation.
    :return: Approximated quantile.
    """
    rng = np.random.default_rng()

    increments = (rng.standard_normal(size=(n_repetitions, n_grid))
                  / np.sqrt(n_grid))
    increments[:, 0] = 0
    W = np.cumsum(increments, axis=-1)

    lambda_val = np.arange(1, n_lambda) / n_lambda
    lambda_ind = (lambda_val * n_grid - 1).astype(int)
    B = W[:, -1] / np.mean(
        np.abs(W[:, lambda_ind] - lambda_val[None, :] * W[:, -1, None]), axis=-1
    )

    p_value = np.mean(test_statistic < B)

    return p_value


if __name__ == '__main__':
    n = 100
    # x = np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.randn(n) / 4
    # x = np.random.randn(n) / 4
    x = np.linspace(0, 1, n) + np.random.randn(n) / 4

    result = kernel_based_tests(x)
    print(result)
