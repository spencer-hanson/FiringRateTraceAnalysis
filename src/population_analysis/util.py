from typing import Callable, Any

import numpy as np
import math


def calc_num_bins(arr):
    q75, q25 = np.percentile(arr, [72, 25])
    iqr = q75 - q25 + 0.0000001
    bins = math.ceil((np.max(arr) - np.min(arr) + 0.000001) / (2 * iqr * np.power(len(arr), -1 / 3)))
    if bins > len(arr):
        bins = int(.5 * len(arr))
    return bins


def filter_by_unitlabel(data: np.ndarray, unit_labels: np.ndarray, label_num: float) -> np.ndarray:
    # assumes unit_labels is array of n x 1 eg [[nan], [nan], [1], ..]
    non_nan = np.logical_not(np.isnan(unit_labels[:, 0]))
    data = data[non_nan]
    unit_labels = unit_labels[non_nan]
    data2 = data[unit_labels[:, 0] == label_num]
    return data2


def make_colormap(func: Callable[[int], Any], size: int):
    class C(object):
        N = size

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)
    return C()
