#
# tests/self_normalization.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-15
# Author: Florian Heinrichs
#
# Proposed test, based on self-normalization of the CUSUM-statistic

import numpy as np


def test_null_function(X: np.ndarray,
                       alpha: float = 0.05,
                       quantile: float = None) -> np.ndarray | bool:
    """
    Test whether the mean function is constantly zero.

    :param X: Observed time series as NumPy array of size (n_time,) or
        (n_time, n_ts).
    :param alpha: Level of test.
    :param quantile: Approximated quantile of test. If None, it is estimated
        during calculation.
    :return: Test decision (True if null hypothesis is rejected).
    """
    expand_dims = len(X.shape) == 1

    if expand_dims:
        X = np.expand_dims(X, axis=-1)

    n = len(X)
    bn = int(n ** (1 / 2 - 1 / 8))

    S = calculate_double_cusum_process(X, bn)

    weights = np.expand_dims(np.arange(1, n + 1) / n, axis=-1)
    numerator = np.max(np.abs(S[-1]), axis=0)
    denom = np.max(np.abs(S[:, -1] - weights * S[-1:, -1]), axis=0)

    if quantile is None:
        quantile = estimate_quantile(alpha, hypothesis='null')

    test_decision = numerator / denom > quantile

    if expand_dims:
        test_decision = test_decision[..., 0]

    return test_decision


def test_constant_function(X: np.ndarray,
                           alpha: float = 0.05,
                           quantile: float = None,
                           t0: float = 1/2,
                           t1: float = 3/4) -> np.ndarray | bool:
    """
    Test whether the mean function is constant.

    :param X: Observed time series as NumPy array of size (n_time,) or
        (n_time, n_ts).
    :param alpha: Level of test.
    :param quantile: Approximated quantile of test. If None, it is estimated
        during calculation.
    :param t0: Free parameter t0 from decision rule.
    :param t1: Free parameter t1 from decision rule.
    :return: Test decision (True if null hypothesis is rejected).
    """
    if quantile is None:
        quantile = estimate_quantile(alpha, hypothesis='const')

    ratio = calculate_test_statistic(X, t0=t0, t1=t1)

    factor = np.sqrt((t0 * (1 - t0)) / ((1 - t1) * (t1 - t0)))
    test_decision = ratio > factor * quantile

    return test_decision


def test_constant_p_value(X: np.ndarray,
                          t0: float = 1 / 2,
                          t1: float = 3 / 4) -> np.ndarray | bool:
    """
    Test whether the mean function is constant, returns p-value rather than a
    test decision.

    :param X: Observed time series as NumPy array of size (n_time,) or
        (n_time, n_ts).
    :param t0: Free parameter t0 from decision rule.
    :param t1: Free parameter t1 from decision rule.
    :return: Test decision (True if null hypothesis is rejected).
    """

    ratio = calculate_test_statistic(X, t0=t0, t1=t1)
    factor = np.sqrt((t0 * (1 - t0)) / ((1 - t1) * (t1 - t0)))
    test_statistic = ratio / factor

    p_value = estimate_p_value(test_statistic)

    return p_value


def calculate_test_statistic(X: np.ndarray,
                             t0: float = 1/2,
                             t1: float = 3/4) -> np.ndarray:
    """
    Calculate test statistic for test of constant function.

    :param X: Observed time series as NumPy array of size (n_time,) or
        (n_time, n_ts).
    :param t0: Free parameter t0 from decision rule.
    :param t1: Free parameter t1 from decision rule.
    :return: Calculated test statistic (ratio).
    """
    expand_dims = len(X.shape) == 1

    if expand_dims:
        X = np.expand_dims(X, axis=-1)

    n = len(X)
    bn = int(n ** (1 / 2 - 1 / 8))
    ln = int(n // bn)

    S = calculate_double_cusum_process(X, bn)

    weights = np.expand_dims(np.arange(1, n + 1) / n, axis=-1)
    S_t0 = S[int(t0 * n)]
    S_t1 = S[int(t1 * n)]
    S_1 = S[-1]

    V = np.sqrt(n) * (np.cumsum(S_t0, axis=0) / n - weights / 2 * S_t0)
    coeff = ((int(t1 * n // ln) - int(t0 * n // ln))
              / (int(n // ln) - int(t0 * n // ln)))
    H_tilde = np.sqrt(n) * (S_t1 - S_t0 - coeff * (S_1 - S_t0))
    H = (np.cumsum(H_tilde, axis=0) / n - weights / 2 * H_tilde)

    numerator = np.max(np.abs(V), axis=0)
    denom = np.max(np.abs(H), axis=0)
    ratio = numerator / denom

    if expand_dims:
        ratio = ratio[..., 0]

    return ratio


def calculate_double_cusum_process(X: np.ndarray, bn: int) -> np.ndarray:
    """
    Calculate the double cusum process S_n(t, s) from X.

    :param X: Observed time series as NumPy array of size (n_time,) or
        (n_time, n_ts).
    :param bn: Block length.
    :return: Double CUSUM process S_n(t, s) as NumPy array.
    """
    expand_dims = len(X.shape) == 1

    if expand_dims:
        X = np.expand_dims(X, axis=-1)

    n_time, n_ts = X.shape
    ln = int(n_time // bn)
    x1 = np.arange(n_time)
    x2 = ((np.arange(ln) * bn)[None, :] + np.arange(bn)[:, None]).flatten()
    x2 = np.concatenate((x2, np.arange(ln * bn, n_time)), axis=0)

    M = np.zeros((n_time, n_time, n_ts))
    M[x1, x2] = X[x2]
    S = M.cumsum(axis=0).cumsum(axis=1) / n_time

    if expand_dims:
        S = S[..., 0]

    return S


def estimate_quantile(alpha: float,
                      n_repetitions: int = 1000,
                      n_grid: int = 1000,
                      hypothesis: str = 'const') -> float:
    """
    Estimate quantile for the test of a null function.

    :param alpha: Level of test.
    :param n_repetitions: Number of trajectories used for the estimation.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param hypothesis: Indicating hypothesis (either 'null' or 'const').
    :return: Estimated quantile.
    """
    ratio = simulate_ratio(n_repetitions=n_repetitions,
                           n_grid=n_grid,
                           hypothesis=hypothesis)
    quantile = np.sort(ratio)[int(n_repetitions * (1 - alpha)) - 1]

    return quantile


def estimate_p_value(test_statistic: float | np.ndarray,
                     n_repetitions: int = 1000,
                     n_grid: int = 1000,
                     hypothesis: str = 'const') -> np.ndarray:
    """
    Estimate quantile for the test of a null function.

    :param test_statistic: Value of test statistic.
    :param n_repetitions: Number of trajectories used for the estimation.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param hypothesis: Indicating hypothesis (either 'null' or 'const').
    :return: Estimated quantile.
    """
    if isinstance(test_statistic, float):
        test_statistic = np.array([test_statistic])

    ratio = simulate_ratio(n_repetitions=n_repetitions,
                           n_grid=n_grid,
                           hypothesis=hypothesis)

    p_value = np.mean(ratio[:, None] > test_statistic[None, :], axis=0)

    return p_value


def simulate_ratio(n_repetitions: int = 1000,
                   n_grid: int = 1000,
                   hypothesis: str = 'const') -> np.ndarray:
    """
    Simulate limiting distribution, which is the ratio of two suprema.

    :param n_repetitions: Number of simulated ratios.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param hypothesis: Indicating hypothesis (either 'null' or 'const').
    :return: Simulated limiting distribution.
    """
    W = simulate_brownian_motions(n_repetitions=2 * n_repetitions, n_grid=n_grid)
    W1, W2 = W[:n_repetitions], W[n_repetitions:]

    if hypothesis == 'null':
        weights = np.arange(1, n_grid + 1) / n_grid
        B = W2 - weights * W2[:, -1:]
    elif hypothesis == 'const':
        B = W2
    else:
        raise ValueError(f"{hypothesis=} is not a valid hypothesis.")

    ratio = np.max(np.abs(W1), axis=-1) / np.max(np.abs(B), axis=-1)

    return ratio


def simulate_brownian_motions(n_repetitions: int = 1000,
                              n_grid: int = 1000) -> np.ndarray:
    """
    Simulate trajectories of (independent) Brownian motions.

    :param n_repetitions: Number of trajectories.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :return: Simulated trajectories as NumPy array.
    """
    rng = np.random.default_rng()

    increments = (rng.standard_normal(size=(n_repetitions, n_grid))
                  / np.sqrt(n_grid))
    increments[:, 0] = 0
    W = np.cumsum(increments, axis=-1)

    return W


if __name__ == '__main__':
    n = 200
    x1 = np.random.randn(n) / 4
    x2 = np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.randn(n) / 4
    x3 = np.linspace(0, 1, n) + np.random.randn(n) / 4

    for x in [x1, x2, x3]:
        td1 = test_null_function(x)
        td2 = test_constant_function(x)
        print(f"Verwerfe H_0 (mu = 0): {td1}, verwerfe H_0 (mu const.): {td2}")

    X = np.array([x1, x2, x3]).transpose()
    td1 = test_null_function(X)
    td2 = test_constant_function(X)
    print(f"Verwerfe H_0 (mu = 0): {td1}, verwerfe H_0 (mu const.): {td2}")

