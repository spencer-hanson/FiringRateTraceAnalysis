import functools
import math
from typing import Optional

import matplotlib
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from population_analysis.processors.nwb import NWBSessionProcessor
from population_analysis.processors.nwb.unit_filter import UnitFilter

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


def _get_spike_idxs(bbool_counts, unit_number, trial_idxs, unit_filter: Optional[UnitFilter] = None):
    counts = bbool_counts

    if unit_filter is not None:
        counts = counts[unit_filter.idxs(), :, :]
    counts = counts[:, trial_idxs, :][unit_number]

    spike_idxss = []
    for trial in counts:
        spike_idxss.append(np.where(trial)[0])
    return spike_idxss


def avg_raster_plot(nwb_session, name, unit_filter: UnitFilter, trial_idxs, num_units, prefix=""):
    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, axs = plt.subplots(2, 1, figsize=(16, 8), width_ratios=[1], height_ratios=[1, 1])

    # Raster
    if num_units == -1:
        num_units = unit_filter.len()
    # TODO use a polynomial to normalize the distribution of spikes to create a valid heatmap for the raster, with
    # specific number of bins to reduce computational time

    # bins = 40
    # ss = np.sum(bool_counts[unit_filter.idxs()][:, trial_idxs], axis=0)
    # hist = np.histogram(ss, bins=bins)
    #
    # total = np.sum(hist[0])
    # hist_vals = hist[0][np.where(hist[0] != 0)[0]]
    # sums = [np.sum(hist[0][:i]) for i in range(1, bins+1)]  # Skip 1 since it's going to be zero
    # xs = range(bins)
    # ys = sums
    # deg = bins
    #
    # poly = np.polynomial.polynomial.Polynomial.fit(xs, ys, deg)
    # inv_poly = np.polynomial.polynomial.Polynomial.fit(ys, xs, deg)
    # dig = np.digitize(ss, hist[1])

    # fig, axs = plt.subplots(2, 1)
    # for j in range(1, bins):  # Skip 0 since only no spikes will give that value
    #     print(f"{j}/{bins}")
    #     jj = []
    #     for i in range(ss.shape[0]):
    #         jj.append(np.where(dig[i] == j)[0])
    #     axs[0].eventplot(jj, colors="black", lineoffsets=1, linelengths=1, alpha=round(np.sum(hist[0][:j])/total, 2))
    #
    # fig.show()

    for uidx in range(num_units):
        print(f"{uidx}/{num_units} ", end="")
        spike_idxs = _get_spike_idxs(bool_counts, uidx, trial_idxs, unit_filter=unit_filter)
        axs[0].eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1, alpha=.2)

    print("")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Trial #")

    # Mean response
    units = nwb_session.units()[unit_filter.idxs(), :, :][:, trial_idxs, :]  # units,trials,t
    avgd_units = np.mean(np.mean(units, axis=1), axis=0)  # average over trials and units
    axs[1].plot(avgd_units)
    axs[1].set_xlabel("Time (binned by 20ms)")
    axs[1].set_ylabel("Firing rate (spikes/20ms)")

    # Fig stuff
    fig.suptitle(f"{name} - Averaged over all units, mean response over trials and units")
    fig.savefig(f"{prefix}{name}_avg_{num_units}.png")
    fig.show()
    tw = 2


def multi_raster_plot(nwb_session, name_and_trial_idxs, absolute_unit_number, unit_filter: UnitFilter, suppress_passing_filename_suffix=False):
    # Graph a single unit's specific response type and mean response
    # name_and_trial_idxs is a tuple like (name, trial_idxs)
    # passing_func(unit_num) -> bool if unit passes filtering, will change title
    # suppress_passing_filename_suffix = True will remove the suffix '- passing'

    ncols = len(name_and_trial_idxs)
    nrows = 2  # Top row is response of unit, bottom is mean response of all trials

    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(16, 8),
                            width_ratios=[1]*ncols, height_ratios=[1]*nrows)
    axs = axs.reshape((nrows, ncols))

    count = 0
    for c in range(ncols):
        name, trial_idxs = name_and_trial_idxs[count]
        spike_idxs = _get_spike_idxs(bool_counts, absolute_unit_number, trial_idxs)  # Don't filter units here
        axs[0, c].eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)
        axs[0, c].set_title(f"{name}")
        count = count + 1
        responses = nwb_session.units()[:, trial_idxs, :][absolute_unit_number]
        responses = np.mean(responses, axis=0)
        axs[1, c].plot(responses)
        axs[1, c].set_title("Mean response of all trials")

    axs[nrows-1, 0].set_xlabel("Time (ms)")
    axs[0, 0].set_ylabel("Trial #")
    cluster_number = nwb_session.nwb.processing["behavior"]["unit-labels"].data[:][absolute_unit_number]

    passes = "PASSES" if unit_filter.passes_abs(absolute_unit_number) else "FAILS"

    title_str = f"Unit {absolute_unit_number} Cluster {cluster_number}"
    save_name = f"{passes}_multi_u{absolute_unit_number}_c{cluster_number}.png"

    if not suppress_passing_filename_suffix:
        title_str = title_str + " - {passes}"
    else:
        save_name = f"u{absolute_unit_number}_c{cluster_number}.png"

    fig.suptitle(title_str)
    fig.savefig(save_name)
    fig.show()
    plt.close(fig)
    tw = 2


def single_raster_plot(nwb_session, name, trial_idxs, unit_num, unit_filter: Optional[UnitFilter] = None):
    # Plot a single unit's raster plot for a given trial

    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, ax = plt.subplots()

    spike_idxs = _get_spike_idxs(bool_counts, unit_num, trial_idxs, unit_filter=unit_filter)
    ax.eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial #")

    fig.suptitle(f"{name} - Unit {unit_num}")
    fig.savefig(f"{name}_single_u{unit_num}.png")
    # fig.show()
    tw = 2


def mean_response(nwb_session: NWBSessionProcessor, name, unit_filter: UnitFilter, trial_idxs, prefix=""):
    units = nwb_session.units()[unit_filter.idxs(), :, :][:, trial_idxs, :]
    avgd_units = np.mean(units, axis=1)

    mean_response_custom(avgd_units, name, prefix=prefix)


def mean_response_custom(averaged_units, name, prefix=""):
    fig, ax = plt.subplots()

    for u in averaged_units:
        ax.plot(u)
    fig.suptitle(f"{name} mean response across trials for all units")
    ax.set_xlabel("Time (binned by 20ms)")
    ax.set_ylabel("Firing rate (spikes/20ms)")
    fig.savefig(f"{prefix}{name}_mean_response.png")
    fig.show()
    tw = 2


def standard_multi_rasters(sess: NWBSessionProcessor, unit_filter: UnitFilter, suppress_passing_filename_suffix=False):
    # todo only_passing? change loop to idxs, use unit_filter.idxs(), default to all idxs
    total = sess.num_units
    for unum in range(total):
        print(f"Processing unit {unum}/{total}")
        multi_raster_plot(
            sess,
            name_and_trial_idxs=[
                ("Rp_Extra", sess.probe_trial_idxs),
                ("Rs", sess.saccade_trial_idxs),
                ("Rmixed", sess.mixed_trial_idxs)
            ],
            unit_filter=unit_filter,
            absolute_unit_number=unum,
            suppress_passing_filename_suffix=suppress_passing_filename_suffix
        )


def standard_all_summary(sess):
    matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    passing_unit_filter = sess.qm_unit_filter().append(
        sess.probe_zeta_unit_filter()
    )

    print("Plotting mean responses..")
    mean_response(sess, "Rs - Passing", passing_unit_filter, sess.saccade_trial_idxs)
    mean_response(sess, "Rp_Extra - Passing", passing_unit_filter, sess.probe_trial_idxs)
    mean_response(sess, "Rmixed - Passing", passing_unit_filter, sess.mixed_trial_idxs)
    mean_response_custom(np.mean(sess.rp_peri_units()[passing_unit_filter.idxs(), :, :], axis=1), "Rp_Peri")

    print("Plotting avg rasters..")
    avg_raster_plot(sess, "Rs", passing_unit_filter, sess.saccade_trial_idxs, -1)
    avg_raster_plot(sess, "Rmixed", passing_unit_filter, sess.mixed_trial_idxs, -1)
    avg_raster_plot(sess, "Rp_Extra", passing_unit_filter, sess.probe_trial_idxs, -1)

    print("Starting on mult-raster plots..")
    standard_multi_rasters(sess, passing_unit_filter)

    print("Done!")


def main():
    filename = "2023-05-15_mlati7_output"
    matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSessionProcessor("../scripts", filename, "../graphs")
    # Create the mean responses, avg raster plot, and individual unit rasters for the standard filter
    # standard_all_summary(sess)

    # This will create a multi-raster of all units and save them in the form u<unit_num>_c<cluster_num>
    standard_multi_rasters(sess, UnitFilter.empty(sess.num_units), suppress_passing_filename_suffix=True)

    # Baseline of timepoints
    # units_baseline_firingrate(mixed_units)
    # units_baseline_firingrate(probe_units)
    # units_baseline_firingrate(saccade_units)

    # Mean responses
    # mean_response(sess, "Rs", activity_idxs, sess.saccade_trial_idxs)
    # mean_response(sess, "Rp_Extra", activity_idxs, sess.probe_trial_idxs)
    # mean_response(sess, "Rmixed", activity_idxs, sess.mixed_trial_idxs)
    # mean_response_custom(np.mean(sess.rp_peri_units()[activity_idxs, :, :], axis=1), "Rp_Peri")

    # Average of all units in one raster
    # avg_raster_plot(sess, "Rs", activity_idxs, sess.saccade_trial_idxs, 1000)
    # avg_raster_plot(sess, "Rmixed", activity_idxs, sess.mixed_trial_idxs, 1000)
    # avg_raster_plot(sess, "Rp_Extra", activity_idxs, sess.probe_trial_idxs, 1000)

    tw = 2


if __name__ == "__main__":
    main()
