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

from population_analysis.util import calc_num_bins


def graph_dists(dists, original, name):
    bins = calc_num_bins(dists)
    hist = np.histogram(dists, bins=bins, density=True)
    bar_y = hist[0] / np.sum(hist[0])
    plt.title(f"Probability Density of Quantification Value {name}")
    plt.bar(range(len(hist[0])), bar_y, label="value probability")

    tick_width = 10
    x_labels = np.round(hist[1][0::tick_width], decimals=5)
    x_ticks = list(range(0, len(hist[0]), tick_width))
    x_labels = x_labels[:len(x_ticks)]

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
    # func1 = lambda x: 1.5*x
    func1 = lambda x: np.random.normal()
    func2 = lambda x: np.random.normal(loc=0.5)
    # func2 = lambda x: 1.5*x

    jitter = lambda x: 0
    # jitter = lambda x: random.uniform(0, 2) * 1 if random.randint(0, 1) % 2 else -1

    class1 = [[func1(x)+jitter(x) for x in range(35)] for _ in range(10)]
    class2 = [[func2(x)+jitter(x) for x in range(35)] for _ in range(10)]
    """
    TODO
    Take each neuron fr as a dimension, use 100ms after the probe and average the fr, concat for all neurons
    
    
    """
    # plt.plot(range(35), class1[0], color="red")
    # plt.plot(range(35), class2[0], color="green")
    # plt.show()
    return class1, class2


def _make_sanity_datatset(data):
    np.random.shuffle(data)
    split_percentage = 0.5
    split_len = int(len(data)*split_percentage)
    return [data[:split_len], data[split_len:]]


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

    probe_units = nwb.units["trial_response_firing_rates"].data[:, probe_trial_idxs]
    saccade_units = nwb.units["trial_response_firing_rates"].data[:, saccade_trial_idxs]

    # TODO Do we concat trials? idk doing that currently
    probe_unit_timepoints = probe_units.swapaxes(0, 2)[0]
    saccade_unit_timepoints = saccade_units.swapaxes(0, 2)[0]

    flattened_probe_units = probe_units.reshape((-1, 35))
    flattened_saccade_units = saccade_units.reshape((-1, 35))

    quans_to_run = [
        # Test Quan
        # (*TestQuantification.DATA, TestQuantification()),
        # (*TestQuantification.DATA, SlowQuantification()),

        # Sanity check the sanity check
        # (*_create_test_data(), EuclidianQuantification("sanity")),

        # Quantification between R_p(Extra) and R_s  (sanity check that there should be a difference)
        (probe_units, saccade_units, EuclidianQuantification("sanity2")),

        # Sanity check that there should be no difference between same 'cloud'
        # (probe_units, probe_units, EuclidianQuantification("probe")),
        # (saccade_units, saccade_units, EuclidianQuantification("saccade")),
        # (*_make_sanity_datatset(np.copy(saccade_unit_timepoints)), EuclidianQuantification("SaccadeSanity")),
    ]

    for quan_params in quans_to_run:
        quan_dist = QuanDistribution(*quan_params)
        calculated_dists = {"dists": quan_dist.calculate(), "original": quan_dist.original(), "name": quan_dist.get_name()}

        # now = pendulum.now()
        # fp = open(f"{quan_dist.get_name()}-dists-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}.json", "w")
        # json.dump(calculated_dists, fp)
        # fp.close()

        graph_dists(**calculated_dists)


if __name__ == "__main__":
    main()
