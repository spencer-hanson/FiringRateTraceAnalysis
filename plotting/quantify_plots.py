import json
import math
import sys
import random

import numpy as np
import pendulum
from pynwb import NWBHDF5IO

from population_analysis.quantification import QuanDistribution, TestQuantification, SlowQuantification
from population_analysis.quantification.euclidian import EuclidianQuantification

import matplotlib.pyplot as plt


def _filter_out_zeros(arr):
    epsilon = 0.00001
    sums = np.sum(arr, axis=1)
    filtered_idxs = np.where(np.logical_not(sums < epsilon))
    val = arr[filtered_idxs]
    return val


def _calc_num_bins(arr):
    q75, q25 = np.percentile(arr, [72, 25])
    iqr = q75 - q25 + 0.0000001
    bins = math.ceil((np.max(arr) - np.min(arr) + 0.000001) / (2 * iqr * np.power(len(arr), -1 / 3)))
    return bins


def graph_dists(dists, original):
    bins = _calc_num_bins(dists)
    hist = np.histogram(dists, bins=bins, density=True)
    bar_y = hist[0] / np.sum(hist[0])
    plt.title("Probability Density of Quantification Value")
    plt.bar(range(len(hist[0])), bar_y, label="value probability")

    tick_width = 10
    x_labels = np.round(hist[1][0::tick_width], decimals=5)
    x_ticks = range(0, len(hist[0]), tick_width)

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=90)
    plt.subplots_adjust(bottom=0.25)  # Add 25% space to the bottom for the label size
    # To find which bin original is in - the first entry of where the original < hist bins
    line_x = np.where(original < hist[1])[0]
    if not line_x.size > 0:
        line_x = original / (hist[1][1] - hist[1][0])
    else:
        line_x = line_x[0]
    plt.vlines(line_x, np.min(bar_y), np.max(bar_y), linestyles="dashed", color="red", label="original value")
    plt.legend()
    plt.show()


def _create_test_data():
    func1 = lambda x: 1.5*x
    func2 = lambda x: 1.6*x
    jitter = lambda x: random.uniform(0, 2) * 1 if random.randint(0, 1) % 2 else -1

    class1 = [[func1(x)+jitter(x) for x in range(35)] for _ in range(10)]
    class2 = [[func2(x)+jitter(x) for x in range(35)] for _ in range(10)]

    # plt.plot(range(35), class1[0], color="red")
    # plt.plot(range(35), class2[0], color="green")
    # plt.show()
    return class1, class2


def main():
    _create_test_data()

    # dist_json_filepath = "saccade-dists-3-18_15-58-57.json"
    # dist_json_filepath = "probe-dists-3-18_15-1-38.json"
    # dist_fp = open(dist_json_filepath, "r")
    # dist_data = json.load(dist_fp)
    # graph_dists(dist_data["dists"], dist_data["original"])
    filepath = "../scripts/2023-05-15_mlati7_output.nwb"

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    probe_trial_idxs = nwb.processing["behavior"]["unit-trial-probe"].data[:]
    saccade_trial_idxs = nwb.processing["behavior"]["unit-trial-saccade"].data[:]

    probe_units = nwb.units["trial_firing_rates"].data[:, probe_trial_idxs]
    saccade_units = nwb.units["trial_firing_rates"].data[:, saccade_trial_idxs]

    # Do we concat trials? idk doing that currently
    probe_units = probe_units.reshape((-1, 35))
    saccade_units = saccade_units.reshape((-1, 35))

    probe_units = _filter_out_zeros(probe_units)
    saccade_units = _filter_out_zeros(saccade_units)

    quans_to_run = [
        # Test Quan
        # (*TestQuantification.DATA, TestQuantification()),
        # (*TestQuantification.DATA, SlowQuantification()),

        # Sanity check the sanity check
        (*_create_test_data(), EuclidianQuantification("sanity")),

        # Quantification between R_p(Extra) and R_s  (sanity check that there should be a difference)
        # (probe_units, saccade_units, EuclidianQuantification()),

        # Sanity check that there should be no difference between same 'cloud'
        # (probe_units, probe_units, EuclidianQuantification("probe")),
        # (saccade_units, saccade_units, EuclidianQuantification("saccade")),
    ]

    for quan_params in quans_to_run:
        quan_dist = QuanDistribution(*quan_params)
        org = quan_dist.original()
        calculated_dists = {"dists": quan_dist.calculate(), "original": quan_dist.original()}

        now = pendulum.now()
        fp = open(f"{quan_dist.get_name()}-dists-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}.json", "w")
        json.dump(calculated_dists, fp)
        fp.close()

        graph_dists(**calculated_dists)


if __name__ == "__main__":
    main()
