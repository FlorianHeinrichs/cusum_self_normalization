#
# data_generation.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-20
# Author: Florian Heinrichs
#
# Script to generate data, data generating processes based on the paper:
# - Heinrichs, F., & Dette, H. (2021). A distribution free test for changes in
#   the trend function of locally stationary processes. Electronic Journal of
#   Statistics, 15(2), 3762-3797.

import numpy as np


def generate_data(n: int, n_samples: int, mean: str,
                  error: str, sigma: str, std: float = 1.0) -> np.ndarray:
    """
    Wrapper function to generate noisy data.

    :param n: Length of generated time series.
    :param n_samples: Number of generated time series.
    :param mean: Type of mean in ['1', '2', 'abrupt', 'const'].
    :param error: Type of error in ['iid', 'ar', 'ma'].
    :param sigma: Type of sigma in ['0', '1', '2', '3'].
    :param std: Standard deviation of i.i.d. innovations.
    :return: NumPy array containing noisy data.
    """
    if mean == '1':
        mean = mu_1(n)
    elif mean == '2':
        mean = mu_2(n)
    elif mean == 'abrupt':
        mean = mu_abrupt(n)
    elif mean == '1-':
        mean = 0.5 - mu_1(n)
    elif mean == '2-':
        mean = 1.5 - mu_2(n)
    elif mean == 'abrupt-':
        mean = 1 - mu_abrupt(n)
    elif mean == 'const':
        mean = np.zeros((n, 1))
    else:
        raise ValueError("Mean type unknown.")

    if error == 'iid':
        error = generate_iid(n, n_samples, std=std)
    elif error == 'ar':
        error = generate_ar(n, n_samples, std=std)
    elif error == 'ma':
        error = generate_ma(n, n_samples, std=std)
    else:
        raise ValueError("Error type unknown.")

    if sigma == '0':
        sigma = np.ones((n, 1))
    elif sigma == '1':
        sigma = sigma_1(n)
    elif sigma == '2':
        sigma = sigma_2(n)
    elif sigma == '3':
        sigma = sigma_3(n)
    else:
        raise ValueError("Sigma type unknown.")

    return mean + sigma * error


def mu_1(n: int, a: float = 2.) -> np.ndarray:
    """
    Generates a non-monotonically decreasing function (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :param a: Scaling parameter for quadratic term.
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    mu = np.sin(8 * np.pi * x) / 2
    mu[n // 4:] += a * (x[n // 4:] - 1 / 4) ** 2

    return np.expand_dims(mu, axis=1)


def mu_2(n: int) -> np.ndarray:
    """
    Generates a monotonically decreasing function (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    mu = - 3 / 2 * np.sin(2 * np.pi * x) + 1 / 2
    mu[:n // 4] = -1
    mu[(3 * n) // 4:] = 2

    return np.expand_dims(mu, axis=1)


def mu_abrupt(n: int, cp: float = 0.5) -> np.ndarray:
    """
    Generates a step function with change point at "cp" (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :param cp: Position of change point in the unit interval (as float).
    :return: Function values at points np.arange(n) / n
    """
    mu = np.zeros((n, 1))
    mu[int(cp * n):] = 1

    return mu


def generate_iid(n: int, n_samples: int, std: float = 1.0) -> np.ndarray:
    """
    Generates i.i.d. errors according to the specified distribution.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param std: Standard deviation of random variables.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    rng = np.random.default_rng()
    errors = rng.normal(loc=0, scale=1, size=(n, n_samples)) * std

    return errors


def generate_ar(n: int, n_samples: int, burn_in: int = 100,
                a: float = 0.5, std: float = 1.0) -> np.ndarray:
    """
    Generates AR(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param burn_in: Time steps used for burn in of AR process.
    :param a: Autoregressive coefficient.
    :param std: Standard deviation of i.i.d. innovations.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    epsilon = generate_iid(n + burn_in, n_samples, std=std)
    errors = np.zeros((n + burn_in, n_samples))

    errors[0] = epsilon[0]
    for i in range(1, n + burn_in):
        errors[i] = a * errors[i - 1] + epsilon[i]

    var_errors = 1 / (1 - a ** 2)
    errors = errors[burn_in:] / np.sqrt(var_errors)

    return errors


def generate_ma(n: int, n_samples: int, a: float = 0.5,
                std: float = 1.0) -> np.ndarray:
    """
    Generates MA(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param a: MA coefficient.
    :param std: Standard deviation of i.i.d. innovations.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    epsilon = generate_iid(n + 1, n_samples, std=std)
    errors = epsilon[1:] + a * epsilon[:-1]

    var_errors = 1 + a ** 2
    errors = errors / np.sqrt(var_errors)

    return errors


def sigma_1(n: int) -> np.ndarray:
    """
    Generates a monotonically increasing function (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    sigma = 1 / 2 + x

    return np.expand_dims(sigma, axis=1)


def sigma_2(n: int) -> np.ndarray:
    """
    Generates a non-monotonic function (defined on the unit interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    sigma = 1 - np.cos(2 * np.pi * x) / 2

    return np.expand_dims(sigma, axis=1)


def sigma_3(n: int) -> np.ndarray:
    """
    Generates a step function (defined on the unit interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    sigma = np.ones((n, 1)) / 2
    sigma[n // 2:] = 3 / 2

    return sigma


def display_mu(n: int = 200):
    import matplotlib.pyplot as plt
    from tueplots import bundles, figsizes

    plt.rcParams.update(bundles.icml2022(family="Times New Roman", usetex=False))
    plt.rcParams.update(figsizes.icml2022_full())

    x = np.linspace(0, 1, n)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(x, mu_1(n), label='$\mu_1$', c='black')
    axes[0].plot(x, mu_2(n), label='$\mu_2$', c='gray', ls='dashed')
    axes[0].plot(x, mu_abrupt(n), label='$\mu_3$', c='darkgray', ls='dashdot')

    axes[1].plot(x, 0.5 - mu_1(n), label='$\mu_4$', c='black')
    axes[1].plot(x, 1.5 - mu_2(n), label='$\mu_5$', c='gray', ls='dashed')
    axes[1].plot(x, 1 - mu_abrupt(n), label='$\mu_6$', c='darkgray', ls='dashdot')

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()


def display_examples(n: int = 200):
    import matplotlib.pyplot as plt
    from tueplots import bundles, figsizes

    plt.rcParams.update(bundles.icml2022(family="Times New Roman", usetex=False))
    plt.rcParams.update(figsizes.icml2022_full())

    fig, axes = plt.subplots(1, 2)

    for i, (ax, sigma) in enumerate(zip(axes, [sigma_2, sigma_3])):
        error = generate_ar(n, 1)
        y = sigma(n) * error
        ax.plot(y, c='gray')
        ax.set_title(f"$\sigma_{i+2}$")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    display_examples()