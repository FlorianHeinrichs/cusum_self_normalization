#
# case_study.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-22
# Author: Florian Heinrichs
#
# Main script containing real data experiments.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles, figsizes

from tests.bootstrap_test import bootstrap_test
from tests.lle_based_tests import kernel_based_tests
from tests.lrv_based_tests import lrv_based_test
from tests.self_normalization import test_constant_p_value

plt.rcParams.update(bundles.icml2022(family="Times New Roman", usetex=False))
plt.rcParams.update(figsizes.icml2022_full())


def load_data(city: str, mode: str = 'min') -> pd.DataFrame:
    codes = {'Boulia Airport': '038003',
             'Gayndah Post Office': '039039',
             'Gunnedah Pool': '055023',
             'Hobart TAS': '094029',
             'Melbourne Regional Office': '086071',
             'Cape Otway Lighthouse': '090015',
             'Robe': '026026',
             'Sydney': '066062'}

    code = codes[city]

    if mode == 'min':
        filepath = f"data/min_temp/IDCJAC0011_{code}_1800_Data.csv"
        column = "Minimum temperature (Degree C)"
    else:
        filepath = f"data/max_temp/IDCJAC0010_{code}_1800_Data.csv"
        column = "Maximum temperature (Degree C)"
    df = pd.read_csv(filepath)
    data = df.iloc[:, 2:6]
    data.rename(columns={column: 'Temperature'}, inplace=True)

    return data


def experiment(mode: str = 'min', month: int = 7) -> dict:
    """
    Main function for running the experiments.

    :param mode: Using either daily minimum ('min') or maximum ('max')
        temperatures.
    :param month: Month to use, given as integer.
    :return: p-values for each time series.
    """

    cities = ['Boulia Airport', 'Gayndah Post Office', 'Gunnedah Pool',
              'Hobart TAS', 'Melbourne Regional Office',
              'Cape Otway Lighthouse', 'Robe', 'Sydney']

    p_values = {}

    for city in cities:
        data = load_data(city, mode=mode)
        mean_temp = data[data['Month'] == month].groupby(
            'Year')['Temperature'].mean().dropna().to_numpy()[::-1]
        lle_based = kernel_based_tests(mean_temp, return_p_value=True)[0]
        p_values[city] = (lle_based['Rel_Dev_Gumbel'][0],
                          lle_based['Rel_Dev_Gauss'][0],
                          lle_based['L2_Self_Norm'][0],
                          bootstrap_test(mean_temp, return_p_value=True)[1],
                          lrv_based_test(mean_temp, return_p_value=True)[1],
                          test_constant_p_value(mean_temp, t0=1/3, t1=2/3)[0],
                          test_constant_p_value(mean_temp, t0=1/3, t1=1/2)[0],
                          # test_constant_p_value(mean_temp, t0=1/4, t1=3/8)[0],
                          )

    return p_values


def plot_data(mode: str = 'min', month: int = 7):
    """
    Main function for running the experiments.

    :param mode: Using either daily minimum ('min') or maximum ('max')
        temperatures.
    :param month: Month to use, given as integer.
    """
    cities = ['Gayndah Post Office', 'Robe', 'Sydney']
    fig, axes = plt.subplots(1, 3)

    for ax, city in zip(axes, cities):
        data = load_data(city, mode=mode)
        mean_temp = data[data['Month'] == month].groupby(
            'Year')['Temperature'].mean().dropna().to_numpy()
        index = data[data['Month'] == month].groupby(
            'Year')['Temperature'].mean().dropna().index.to_numpy()
        ax.plot(index, mean_temp, label=city, c='gray')
        ax.plot(index, np.ones_like(index) * np.mean(mean_temp), c='black', ls='dashed')

        ax.set_title(city)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p_values = experiment(mode='min')
    for city, values in p_values.items():
        print(f"{city}: {' - '.join([str(v) for v in values])}")
