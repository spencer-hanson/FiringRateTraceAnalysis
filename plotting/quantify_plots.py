import json
import math

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
    plt.bar(range(len(hist[0])), hist[0] / np.sum(hist[0]))

    tick_width = 10
    x_labels = np.round(hist[1][0::tick_width], decimals=5)
    x_ticks = range(0, len(hist[0]), tick_width)

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=90)
    plt.subplots_adjust(bottom=0.25)  # Add 25% space to the bottom for the label size
    plt.show()


def main():
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
        # (*TestQuantification.DATA, TestQuantification())
        # (*TestQuantification.DATA, SlowQuantification())

        # Quantification between R_p(Extra) and R_s  (sanity check that there should be a difference)
        # (probe_units, saccade_units, EuclidianQuantification()),

        # Sanity check that there should be no difference between same 'cloud'
        (probe_units, probe_units, EuclidianQuantification("probe")),
        (saccade_units, saccade_units, EuclidianQuantification("saccade")),
    ]

    for quan_params in quans_to_run:
        quan_dist = QuanDistribution(*quan_params)
        calculated_dists = quan_dist.calculate()

        now = pendulum.now()
        fp = open(f"{quan_dist.get_name()}-dists-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}.json", "w")
        json.dump(calculated_dists, fp)
        fp.close()

        graph_dists(calculated_dists, quan_dist.original())
    # quan_dist = QuanDistribution(probe_units, saccade_units, quantification())
    # graph_dists(quan_dist.calculate())

    """
    
    # Scaled out version to include original value
    v = EuclidianQuantification().calculate(probe_units, saccade_units)
    dists2 = list(dists)
    dists2.append(v)
    
    bins = _calc_num_bins(dists2)
    hist = np.histogram(dists2, bins=bins, density=True)
    plt.bar(range(len(hist[0])), hist[0]/np.sum(hist[0]))
    plt.xticks(ticks=range(0, len(hist[0]), 1000), labels=np.round(hist[1][0::1000], decimals=5), rotation=90)
    plt.subplots_adjust(bottom=0.25)
    plt.vlines(len(hist[0]), 0, .01, color="red", label="Separated Value", linestyles="dashed")
    plt.legend()
    plt.show()
    
    """
    plt.show()


if __name__ == "__main__":
    main()
