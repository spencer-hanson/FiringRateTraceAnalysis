import os

import numpy as np
from pynwb import NWBHDF5IO

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.quantification.euclidian import EuclidianQuantification
import matplotlib.pyplot as plt


def _calc_dists(data_dict):
    # data_dict is {"Rp(Extra)": <rp data>, ..}
    euclid_dist = EuclidianQuantification("Pairwise")
    dists = {}

    for t in range(NUM_FIRINGRATE_SAMPLES):
        data_pairs = list(data_dict.items())
        for i in range(len(data_pairs)):
            name, data = data_pairs.pop()
            for name1, data1 in data_pairs:
                if name == name1:
                    continue
                k = f"{name}-{name1}"
                if k not in dists:
                    dists[k] = []
                dists[k].append(euclid_dist.calculate(
                    data[:, :, t].swapaxes(0, 1),
                    data1[:, :, t].swapaxes(0, 1)
                ))
    return dists


def calculate_pairwise_mean_distances(data_dict):
    # Distance between the means of each data type (probe, saccade, mixed, etc..) over time

    dists = _calc_dists(data_dict)
    for pair_name, vals in dists.items():
        plt.plot(range(NUM_FIRINGRATE_SAMPLES), vals)
        plt.title(pair_name)
        plt.show()
    tw = 2


def main():
    filename = "2023-05-15_mlati7_output"
    filepath = "../scripts/" + filename + ".nwb"
    filename_prefix = f"../graphs/{filename}"
    if not os.path.exists(filename_prefix):
        os.makedirs(filename_prefix)

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    probe_trial_idxs = nwb.processing["behavior"]["unit-trial-probe"].data[:]
    saccade_trial_idxs = nwb.processing["behavior"]["unit-trial-saccade"].data[:]
    mixed_trial_idxs = nwb.processing["behavior"]["unit-trial-mixed"].data[:]

    # Filter out mixed trials that saccades are more than 20ms away from the probe
    mixed_rel_timestamps = nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]
    mixed_filtered_idxs = np.abs(mixed_rel_timestamps) <= 0.02  # 20 ms
    mixed_trial_idxs = mixed_trial_idxs[mixed_filtered_idxs]

    # (units, trials, t)
    probe_units = nwb.units["trial_response_firing_rates"].data[:, probe_trial_idxs]
    saccade_units = nwb.units["trial_response_firing_rates"].data[:, saccade_trial_idxs]
    mixed_units = nwb.units["trial_response_firing_rates"].data[:, mixed_trial_idxs]
    rp_peri_units = nwb.units["r_p_peri_trials"].data[:]
    tw = 2

    data_dict = {
            "Rp(Extra)": probe_units,
            "Rs": saccade_units,
            "Rmixed": mixed_units,
            "Rp(Peri)": rp_peri_units
        }

    calculate_pairwise_mean_distances(
        data_dict
    )


if __name__ == "__main__":
    main()
