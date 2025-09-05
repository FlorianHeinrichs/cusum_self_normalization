#
# experiments.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-20
# Author: Florian Heinrichs
#
# Script containing main experiments for test comparison.

from datetime import datetime
import json
from time import time

import numpy as np

from data_generation import generate_data
from tests.bootstrap_test import bootstrap_test
from tests.lle_based_tests import kernel_based_tests
from tests.lrv_based_tests import lrv_based_test
from tests.self_normalization import (test_null_function, test_constant_function,
                                      estimate_quantile)


def experiment(n: int, n_samples: int, mean: str, error: str, sigma: str,
               std: float = 1.0, alpha: float = 0.05) -> tuple:
    """
    Main function for running the experiments.

    :param n: Length of generated time series.
    :param n_samples: Number of generated time series.
    :param mean: Type of mean in ['1', '2', 'abrupt', 'const'].
    :param error: Type of error in ['iid', 'ar', 'ma'].
    :param sigma: Type of sigma in ['0', '1', '2', '3'].
    :param std: Standard deviation of i.i.d. innovations.
    :param alpha: Level of test.
    :return: Tuple of dictionaries with test decisions for all tests and their
        respective run times.
    """
    # Generate Data
    data = generate_data(n, n_samples, mean, error, sigma, std=std)

    # Quantile Estimation
    quantile_null = estimate_quantile(alpha, hypothesis='null')
    quantile_const = estimate_quantile(alpha, hypothesis='const')

    # Do tests
    test_decisions, times = kernel_based_tests(data, alpha=alpha)

    tests = [
        ('Bootstrap', bootstrap_test, {}),
        ('LRV_based', lrv_based_test, {}),
        ('SN_null', test_null_function, {'quantile': quantile_null}),
        ('SN_const', test_constant_function, {'quantile': quantile_const}),
        ('SN_const_equal', test_constant_function, {'quantile': quantile_const,
                                                    't0': 1/3, 't1': 2/3})
    ]

    for test_name, test, test_args in tests:
        test_times, results = [], []
        for x in data.transpose():
            start = time()
            result = test(x, alpha=alpha, **test_args)
            test_times.append(time() - start)
            results.append(result)

        test_decisions[test_name] = results
        times[test_name] = test_times

    return test_decisions, times


def get_config(n_samples) -> list:
    """
    Get list of configurations for experiments.

    :param n_samples: Number of generated time series.
    :return: List of configurations, where each entry of the list is a tuple
        of arguments of experiment().
    """
    config = [
        (n, n_samples, mean, error, sigma, std)
        for n in [100, 200, 500, 1000]
        for mean in ['1', '2', 'abrupt', 'const']
        for error in ['iid', 'ar', 'ma']
        for sigma in ['0', '1', '2', '3']
        for std in [1/4, 1/2, 1]
    ]

    return config


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return np.array(data).tolist()
    elif isinstance(data, tuple):
        return [convert_to_serializable(value) for value in data]
    else:
        return data


if __name__ == '__main__':
    n_samples = 2
    config = get_config(n_samples)
    results = {}

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"../results/samples{n_samples}_" + now + ".json"

    for args in config:
        arg_str = f"n{args[0]}_mu{args[2]}_{args[3]}_sigma{args[4]}_std{args[5]}"
        print(f"Starting experiment: {arg_str}")
        results[arg_str] = experiment(*args)

        with open(filepath, 'w') as file:
            result_tmp = convert_to_serializable(results)
            json.dump(result_tmp, file, indent=4)
