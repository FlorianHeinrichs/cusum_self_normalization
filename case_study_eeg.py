#
# case_study_eeg.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-29
# Author: Florian Heinrichs
#
# Main script containing real data experiments.


from eeg_et_benchmark.load_data import load_dataset
import numpy as np
import pandas as pd

from tests.bootstrap_test import bootstrap_test
from tests.lle_based_tests import kernel_based_tests
from tests.lrv_based_tests import lrv_based_test
from tests.self_normalization import test_constant_p_value


def experiment(folder: str) -> dict:
    """
    Main function for running the experiments.

    :param folder: Folder to EEG data (if locally available).
    :return: p-values for each time series.
    """
    task = "level-2-smooth"
    eeg_columns = ['EEG_TP9', 'EEG_AF7', 'EEG_AF8', 'EEG_TP10']
    exclude = ["P002_01", "P004_01"] + [
        f"P0{k}_01" for k in list(range(16, 21)) + list(range(62, 68)) + [79]]

    recordings = load_dataset(task=task, exclude=exclude, folder=folder)
    recordings = recordings[0] + recordings[1]

    p_values = {}

    for i, rec in enumerate(recordings):
        data = np.mean(rec[eeg_columns].to_numpy(), axis=1)
        n_len = int((len(data) // 250) * 250)
        data = np.mean(np.reshape(data[:n_len], (n_len // 250, 250)), axis=1)

        lle_based = kernel_based_tests(data, return_p_value=True)[0]
        p_values[i] = (lle_based['Rel_Dev_Gumbel'][0],
                       lle_based['Rel_Dev_Gauss'][0],
                       lle_based['L2_Self_Norm'][0],
                       bootstrap_test(data, return_p_value=True)[1],
                       lrv_based_test(data, return_p_value=True)[1],
                       test_constant_p_value(data, t0=1/3, t1=2/3)[0],
                       test_constant_p_value(data, t0=1/3, t1=1/2)[0])

    return p_values


if __name__ == '__main__':
    folder = "../../BCI_ET_Benchmark/data/csv_preprocessed"
    p_values = experiment(folder=folder)

    df = pd.DataFrame(p_values).transpose()
    print(df.mean())
