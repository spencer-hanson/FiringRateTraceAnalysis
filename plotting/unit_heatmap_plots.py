import os

from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt

"""
Create a heatmap-like view of the average firing rate of each neuron (trial-averaged)
Useful for visualizing a bunch of neurons all at once
"""


def plot_responses(data_dict):
    num_units = data_dict[list(data_dict.keys())[0]].shape[0]
    # fig, axs = plt.subplots(num_units, len(data_dict.keys()))
    data = []
    uidx = 0
    names = []
    for uname, udata in data_dict.items():
        means = np.mean(udata, axis=1)
        data.append(means)
        names.append(uname)
        # for i, m in enumerate(means):
        #     axs[i, uidx].plot(range(NUM_FIRINGRATE_SAMPLES), m)

        uidx = uidx + 1

    fig, ax = plt.subplots(1, len(names), sharey=True)
    fig.set_figwidth(10)
    fig.set_figheight(40)
    fig.set_dpi(140)

    for idx, name in enumerate(names):
        sorted_data = sorted(data[idx], key=lambda x: np.sum(x))  # TODO

        ax[idx].set_title(names[idx])
        ax[idx].imshow(sorted_data)
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

    data_dict = {
        "Rp(Extra)": probe_units,
        "Rs": saccade_units,
        "Rmixed": mixed_units,
        "Rp(Peri)": rp_peri_units
    }
    plot_responses(data_dict)  # Plot the responses as an image for each unit, averaged over trials, heatmap-style

    # Uncomment for splitting a single response type into multiple sections
    # split_data = probe_units
    # num_split = int(split_data.shape[1] * .1)
    # data_dict = {}
    # idxs = []
    # for i in range(1, 9):
    #     start_idx = (i - 1) * num_split
    #     end_idx = i * num_split
    #     idxs.append((start_idx, end_idx))
    #     data_dict[f"Rp(Extra){i}"] = split_data[:, start_idx:end_idx]
    # plot_responses(data_dict)  # Plot the responses as an image for each unit, averaged over trials, heatmap-style
    tw = 2


if __name__ == "__main__":
    main()
