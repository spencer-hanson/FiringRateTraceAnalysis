import os

from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt

from population_analysis.sessions.saccadic_modulation import NWBSession

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
    if len(names) == 1:
        ax = [ax]
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
    sess = NWBSession("E:\\PopulationAnalysisNWBs\\mlati7-2023-05-15-output\\mlati7-2023-05-15-output.hdf.nwb")

    data_dict = {
        "Rp(Extra)": sess.units()[:, sess.trial_filter_rp_extra().idxs()],
        # "Rs": saccade_units,
        # "Rmixed": mixed_units,
        # "Rp(Peri)": rp_peri_units
    }
    plot_responses(data_dict)  # Plot the responses as an image for each unit, averaged over trials, heatmap-style


if __name__ == "__main__":
    main()
