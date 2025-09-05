#
# tests/bootstrap_test.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-15
# Author: Florian Heinrichs
#
# Test based on bootstrap procedure.

import numpy as np

from .utils import convolve


def bootstrap_test(X: np.ndarray,
                   alpha: float = 0.05,
                   n_repetitions: int = 500,
                   return_p_value: bool = False) -> bool | np.ndarray | tuple:
    """
    Bootstrap procedure for CUSUM-based test .

    :param X: Time series given as NumPy array of shape (n_time,) or
        (n_time, n_ts), where n_time is the length of the time series and n_ts
        the number of observed time series.
    :param alpha: Level of test.
    :param n_repetitions: Number of Bootstrap replications.
    :param return_p_value: Indicator if p-value shall be returned as well.
    :return: Returns test decision (True, if H_0 is rejected).
    """
    expand_dim = len(X.shape) == 1

    if expand_dim:
        X = np.expand_dims(X, axis=1)

    n_time, n_ts = X.shape
    m = int(n_time ** (1 / 3))
    n = int(n_time // 3)

    cumsum = np.cumsum(X, axis=0)
    factor = np.expand_dims(np.arange(1, n_time + 1) / n_time, axis=-1)
    cusum_process = cumsum / n_time - factor * cumsum[-1] / n_time
    cusum_statistic = np.sqrt(n_time) * np.sqrt(np.mean(cusum_process ** 2, axis=0))

    rng = np.random.default_rng()
    random_multipliers = rng.standard_normal(size=(n_time, n_repetitions, n_ts))

    boundary_weights = np.expand_dims(np.arange(1, n + 1), axis=1)
    mu = convolve(X, np.ones((1, 2 * n + 1)) / (2 * n + 1), mode='same')
    mu[:n] = np.cumsum(X[:n], axis=0) / boundary_weights
    mu[-n:] = (np.cumsum(X[:-(n + 1):-1], axis=0) / boundary_weights)[::-1]

    X_centered = np.pad(X - mu, ((0, m - 1), (0, 0)))
    inner_sums = convolve(X_centered, np.ones((1, m)), mode='valid') / np.sqrt(m)
    products = random_multipliers * np.expand_dims(inner_sums, axis=1)
    B = np.cumsum(products, axis=0) / np.sqrt(n_time)

    G = B - factor.reshape((n_time, 1, 1)) * B[-1]
    G_norm = np.sqrt(np.mean(G ** 2, axis=0))

    threshold = np.sort(G_norm, axis=0)[int((1 - alpha) * n_repetitions)]
    test_decision = cusum_statistic > threshold

    if return_p_value:
        p_value = np.mean(G_norm > cusum_statistic[None, :], axis=0)
        if expand_dim:
            test_decision = test_decision[0]
            p_value = p_value[0]

        return test_decision, p_value

    else:
        if expand_dim:
            test_decision = test_decision[0]

        return test_decision


if __name__ == '__main__':
    n = 100
    x1 = np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.randn(n) / 4
    x2 = np.random.randn(n) / 4
    x3 = np.linspace(0, 1, n) + np.random.randn(n) / 4


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3)
    for axs, n_rep in zip(axes, [500, 1000]):
        for ax, x in zip(axs, [x1, x2, x3]):
            ress = []
            for _ in range(10):
                res = bootstrap_tmp(x, n_repetitions=n_rep)
                ress.append(res[int(0.92 * n):int(0.98 * n), 0])

            ress = np.array(ress)
            m = np.mean(ress)
            s = np.std(ress)
            print(m, s)
    # plt.show()


    # result = bootstrap_test(x, n_repetitions=100)
    # print(result)
