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


def main():

    filepath = "../scripts/2023-05-15_mlati7_output.nwb"
    unit_to_view = 0

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    mixed_trial_idxs = nwb.processing["behavior"]["unit-trial-mixed"].data[:]
    mixed_units = nwb.units["trial_response_firing_rates"].data[:, mixed_trial_idxs]
    rel_saccades = nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]

    unit_data = mixed_units[unit_to_view]

    zipped = zip(rel_saccades, unit_data)
    sort = sorted(zipped, key=lambda x: x[0])

    # TODO padding for aligning on saccade?
    # min_pad = math.fabs(math.floor(np.min(rel_saccades) / .2))
    # max_pad = math.ceil(np.max(rel_saccades) / .2)
    # total_pad = int(min_pad) + int(max_pad)
    # total_size = 35 + total_pad
    # xdata = range(total_size)
    xdata = range(35)

    # trials_to_graph = len(mixed_trial_idxs)
    trials_to_graph = 5
    fig, ax = plt.subplots(trials_to_graph, 1, sharex=True)

    for i in range(trials_to_graph):
        ax[i].plot(xdata, sort[i][1] + 0.3*i)
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

    tw = 2


if __name__ == "__main__":
    main()
