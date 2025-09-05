#
# tests/lrv_based_tests.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-15
# Author: Florian Heinrichs
#
# Tests based on (global) long-run variance estimation.

import numpy as np
from scipy.stats import kstwobign

from .utils import estimate_global_lrv


def lrv_based_test(X: np.ndarray, alpha: float = 0.05,
                   return_p_value: bool = False) -> bool | np.ndarray | tuple:
    """
    Standard CUSUM-based test with global long-run variance estimation.

    :param X: Time series given as NumPy array of shape (n_time,) or
        (n_time, n_ts), where n_time is the length of the time series and n_ts
        the number of observed time series.
    :param alpha: Level of test.
    :param return_p_value: Indicator if p-value shall be returned as well.
    :return: Returns test decision (True, if H_0 is rejected).
    """
    n = len(X)
    lrv = estimate_global_lrv(X, 0)
    quantile = kstwobign.ppf(1 - alpha)

    cumsum = np.cumsum(X, axis=0)
    cusum_process = cumsum / n - np.arange(1, n + 1) / n * cumsum[-1] / n
    cusum_statistic = np.max(np.abs(cusum_process))

    test_statistic = np.sqrt(n) * cusum_statistic / np.sqrt(lrv)
    test_decision = test_statistic > quantile

    if return_p_value:
        p_value = 1 - kstwobign.cdf(test_statistic)

        return test_decision, p_value

    else:
        return test_decision



if __name__ == '__main__':
    n = 100
    # x = np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.randn(n) / 4
    # x = np.random.randn(n) / 4
    x = np.linspace(0, 1, n) + np.random.randn(n) / 4

    result = lrv_based_test(x)
    print(result)
