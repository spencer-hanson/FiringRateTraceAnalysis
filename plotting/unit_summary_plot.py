import math

import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from population_analysis.processors.nwb import NWBSessionProcessor

"""
NOT FINISHED
Plot the response waveforms of each mixed trial alongside eachother
TODO add a line where the saccade was, align with time
"""


def unfinished_plot_responses(unit_data, responses_to_plot=None):
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


def units_baseline_firingrate(unit_data):
    # Compare the first 8 timepoints for each unit against the mean of the entire response for each timepoint
    # unit data is (units, trials, t)
    # means = np.mean(unit_data[:, :, :8], axis=2)
    # means = np.mean(unit_data, axis=1)
    take_mean = lambda x, v: np.mean(np.mean(x[:, :, :v], axis=2), axis=1)

    plt.stairs(take_mean(unit_data, 8))
    plt.stairs(take_mean(unit_data, unit_data.shape[-1]))
    plt.show()


def _get_spike_idxs(bbool_counts, unit_num, units_idxs, trial_idxs):
    counts = bbool_counts[units_idxs, :, :][:, trial_idxs, :][unit_num]
    spike_idxss = []
    for trial in counts:
        spike_idxss.append(np.where(trial)[0])
    return spike_idxss


def avg_raster_plot(nwb_session, name, units_idxs, trial_idxs, num_units):
    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, ax = plt.subplots()

    if num_units > len(units_idxs):
        num_units = len(units_idxs)

    for uidx in range(num_units):
        print(f"{uidx}/{num_units} ", end="")
        ax.eventplot(_get_spike_idxs(bool_counts, uidx, units_idxs, trial_idxs), colors="black", lineoffsets=1, linelengths=1, alpha=0.1)
    print("")

    fig.suptitle(name)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial #")
    fig.savefig(f"{name}_avg_{num_units}.png")

    fig.show()
    tw = 2


def multi_raster_plot(nwb_session, name, units_idxs, trial_idxs, nrows, ncols, start_unit=0):
    unit_num = start_unit
    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for r in range(nrows):
        print(f"{r + 1}/{nrows} ", end="")
        for c in range(ncols):
            spike_idxs = _get_spike_idxs(bool_counts, unit_num, units_idxs, trial_idxs)
            axs[r, c].eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)
            unit_num = unit_num + 1
    print("")

    axs[nrows-1, 0].set_xlabel("Time (ms)")
    axs[0, 0].set_ylabel("Trial #")

    fig.suptitle(name)
    fig.savefig(f"{name}_multi_{start_unit}.png")
    fig.show()
    tw = 2


def single_raster_plot(nwb_session, name, units_idxs, trial_idxs, unit_num):
    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, ax = plt.subplots()

    spike_idxs = _get_spike_idxs(bool_counts, unit_num, units_idxs, trial_idxs)
    ax.eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial #")

    fig.suptitle(name)
    fig.savefig(f"{name}_single_u{unit_num}.png")
    # fig.show()
    tw = 2


def mean_response(nwb_session: NWBSessionProcessor, name, unit_idxs, trial_idxs):
    units = nwb_session.units()[unit_idxs, :, :][:, trial_idxs, :]
    avgd_units = np.mean(units, axis=1)

    mean_response_custom(avgd_units, name)


def mean_response_custom(averaged_units, name):
    fig, ax = plt.subplots()

    for u in averaged_units:
        ax.plot(u)
    fig.suptitle(f"{name} mean response across trials for all units")
    ax.set_xlabel("Time (binned by 20ms)")
    ax.set_ylabel("Firing rate (spikes/20ms)")
    fig.savefig(f"{name}_mean_response.png")
    fig.show()
    tw = 2


def main():
    filename = "2023-05-15_mlati7_output"
    sess = NWBSessionProcessor("../scripts", filename, "../graphs")

    probe_units, saccade_units, mixed_units, rp_peri_units = sess.zeta_units()

    # units_baseline_firingrate(mixed_units)
    # units_baseline_firingrate(probe_units)
    # units_baseline_firingrate(saccade_units)

    # avg_raster_plot(sess, "Rs", sess.probe_zeta_idxs(), sess.saccade_trial_idxs, 1000)
    # avg_raster_plot(sess, "Rp_Extra", sess.probe_zeta_idxs(), sess.probe_trial_idxs, 1000)
    # avg_raster_plot(sess, "Rmixed", sess.probe_zeta_idxs(), sess.mixed_trial_idxs, 1000)

    # multi_raster_plot(sess, "Rp_Extra", sess.probe_zeta_idxs(), sess.probe_trial_idxs, 2, 2, start_unit=0)
    for u in range(230):
        print(f"{u}/230")
        # single_raster_plot(sess, "Rp_Extra", sess.probe_zeta_idxs(), sess.probe_trial_idxs, u)
        single_raster_plot(sess, "Rs", sess.probe_zeta_idxs(), sess.saccade_trial_idxs, u)
        single_raster_plot(sess, "Rmixed", sess.probe_zeta_idxs(), sess.mixed_trial_idxs, u)

    # mean_response(sess, "Rs", sess.probe_zeta_idxs(), sess.saccade_trial_idxs)
    # mean_response(sess, "Rp_Extra", sess.probe_zeta_idxs(), sess.probe_trial_idxs)
    # mean_response(sess, "Rmixed", sess.probe_zeta_idxs(), sess.mixed_trial_idxs)
    # mean_response_custom(np.mean(sess.rp_peri_units()[sess.probe_zeta_idxs(), :, :], axis=1), "Rp_Peri")

    tw = 2


if __name__ == "__main__":
    main()
