import os

import numpy as np
from pynwb import NWBHDF5IO

from population_analysis.quantification.euclidian import EuclidianQuantification


def calculate_pairwise_mean_distances(data_dict):
    # data_dict is {"Rp(Extra)": <rp data>, ..}
    euclid_dist = EuclidianQuantification("Pairwise")
    # TODO for each timepoint, rn not what we want
    data_pairs = list(data_dict.items())
    dists = {}
    for i in range(len(data_pairs)):
        name, data = data_pairs.pop()
        for name1, data1 in data_pairs:
            if name == name1:
                continue
            dists[f"{name}-{name1}"] = euclid_dist.calculate(
                data,
                data1
            )
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

    calculate_pairwise_mean_distances(
        {
            "Rp(Extra)": probe_units,
            "Rs": saccade_units,
            "Rmixed": mixed_units,
            "Rp(Peri)": rp_peri_units
        }
    )

    # num_units = probe_units.shape[0]
    # to_pca_units = [probe_units, saccade_units, mixed_units, rp_peri_units]
    # pca_units = [x.swapaxes(0, 2).reshape((-1, num_units)) for x in to_pca_units]
    # pca_units = np.vstack(pca_units)


if __name__ == "__main__":
    main()
