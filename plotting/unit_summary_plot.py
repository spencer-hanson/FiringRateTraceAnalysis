import math

import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
NOT FINISHED
Plot the response waveforms of each mixed trial alongside eachother
TODO add a line where the saccade was, align with time
"""


def plot_responses(unit_data, responses_to_plot=None):
    # response list should be (n, t) where n can be trials or units

    # TODO order by saccade dist?
    # zipped = zip(mixed_rel_timestamps, unit_data)
    # sort = sorted(zipped, key=lambda x: x[0])

    # TODO padding for aligning on saccade?
    # min_pad = math.fabs(math.floor(np.min(rel_saccades) / .2))
    # max_pad = math.ceil(np.max(rel_saccades) / .2)
    # total_pad = int(min_pad) + int(max_pad)
    # total_size = 35 + total_pad
    # xdata = range(total_size)
    xdata = range(35)

    # trials_to_graph = len(mixed_trial_idxs)
    # trials_to_graph = min(50, len(mixed_trial_idxs))

    if responses_to_plot is None:
        responses_to_graph = range(len(unit_data))
    else:
        responses_to_graph = responses_to_plot

    fig, ax = plt.subplots(len(responses_to_graph), 1, sharex=True, figsize=(35, 375), dpi=140)

    if len(responses_to_graph) == 1:
        ax = [ax]

    for i, resp in enumerate(responses_to_graph):
        ax[i].plot(xdata, unit_data[resp] + 0.3*i)
        # num_bins_away = int(round(math.fabs(sort[i][0] / .2)))  # Divide by bin, 200ms
        # total_size - 35 - num_zeros

        # Remove axis lines
        for spine in ["left", "right", "top", "bottom"]:
            ax[i].spines[spine].set_visible(False)

        # Remove ticks
        ax[i].tick_params(axis="x", which="both", bottom=False, top=False)
        ax[i].tick_params(axis="y", which="both", left=False, right=False, labelbottom=False)

        # Remove y labels
        ax[i].set_yticklabels([])

        # ax[i].vlines()
    plt.show()


def unit_starting_firingrate(unit_data):
    # unit data is (units, trials, t)
    # means = np.mean(unit_data[:, :, :8], axis=2)
    # means = np.mean(unit_data, axis=1)
    take_mean = lambda x, v: np.mean(np.mean(x[:, :, :v], axis=2), axis=1)

    plt.stairs(take_mean(unit_data, 8))
    plt.stairs(take_mean(unit_data, unit_data.shape[-1]))
    plt.show()


def main():

    filepath = "../scripts/2023-05-15_mlati7_output.nwb"
    # unit_to_view = 0

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

    # unit_num = 0
    # data = mixed_units[unit_num, :]
    # data = np.mean(data, axis=0)[None, :]
    # plot_responses(data, num_responses_to_plot=5)

    # data = np.mean(mixed_units, axis=1)
    # plot_responses(data, responses_to_plot=range(200, 250))
    # plot_responses(data, responses_to_plot=range(250, data.shape[0]))

    unit_starting_firingrate(mixed_units)
    unit_starting_firingrate(probe_units)
    unit_starting_firingrate(saccade_units)
    tw = 2


if __name__ == "__main__":
    main()
