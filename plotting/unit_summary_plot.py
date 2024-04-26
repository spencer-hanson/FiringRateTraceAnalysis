import functools
import math

import matplotlib
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
        ax.eventplot(_get_spike_idxs(bool_counts, uidx, units_idxs, trial_idxs), colors="black", lineoffsets=1, linelengths=1, alpha=.2)
    print("")

    fig.suptitle(f"{name} - Averaged over all units")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial #")
    fig.savefig(f"AAA{name}_avg_{num_units}.png")

    fig.show()
    tw = 2


def multi_raster_plot(nwb_session, name_and_trial_idxs, units_idxs, unit_number, passing_func=None):
    # Graph a single unit's specific response type and mean response
    # name_and_trial_idxs is a tuple like (name, trial_idxs)
    # passing_func(unit_num) -> bool if unit passes filtering, will change title

    ncols = len(name_and_trial_idxs)
    nrows = 2  # Top row is response of unit, bottom is mean response of all trials

    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(16, 8), width_ratios=[1]*ncols, height_ratios=[1]*nrows)
    axs = axs.reshape((nrows, ncols))

    count = 0
    for c in range(ncols):
        name, trial_idxs = name_and_trial_idxs[count]
        spike_idxs = _get_spike_idxs(bool_counts, unit_number, units_idxs, trial_idxs)
        axs[0, c].eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)
        axs[0, c].set_title(f"{name}")
        count = count + 1
        responses = nwb_session.units()[:, trial_idxs, :][unit_number, :, :]
        responses = np.mean(responses, axis=1)
        axs[1, c].plot(responses)
        axs[1, c].set_title("Mean response of all trials")

    axs[nrows-1, 0].set_xlabel("Time (ms)")
    axs[0, 0].set_ylabel("Trial #")

    title_str = f"Unit {unit_number}"
    save_name = f"multi_u{unit_number}.png"
    if passing_func is not None:
        passes = "PASSES" if passing_func(unit_number) else "FAILS"
        title_str = title_str + " - " + passes
        save_name = passes + "_" + save_name

    fig.suptitle(title_str)
    fig.savefig(save_name)
    # fig.show()
    tw = 2


def single_raster_plot(nwb_session, name, units_idxs, trial_idxs, unit_num):
    bool_counts = nwb_session.nwb.units["trial_spike_flags"]  # units x trials x 700

    fig, ax = plt.subplots()

    spike_idxs = _get_spike_idxs(bool_counts, unit_num, units_idxs, trial_idxs)
    ax.eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial #")

    fig.suptitle(f"{name} - Unit {unit_num}")
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


def standard_all_summary(filename):
    matplotlib.use('Agg')   # Suppress matplotlib window opening

    sess = NWBSessionProcessor("../scripts", filename, "../graphs")

    activity_idxs = sess.probe_zeta_idxs()

    mean_response(sess, "Rs", activity_idxs, sess.saccade_trial_idxs)
    mean_response(sess, "Rp_Extra", activity_idxs, sess.probe_trial_idxs)
    mean_response(sess, "Rmixed", activity_idxs, sess.mixed_trial_idxs)
    mean_response_custom(np.mean(sess.rp_peri_units()[activity_idxs, :, :], axis=1), "Rp_Peri")

    avg_raster_plot(sess, "Rs", activity_idxs, sess.saccade_trial_idxs, 1000)
    avg_raster_plot(sess, "Rmixed", activity_idxs, sess.mixed_trial_idxs, 1000)
    avg_raster_plot(sess, "Rp_Extra", activity_idxs, sess.probe_trial_idxs, 1000)

    # Graph individual units
    passing_func = functools.partial(
        sess.passing_multi,
        [
            sess.passing_zeta,  # zeta test
        ]
    )

    total = len(activity_idxs)
    for unum in range(total):
        print(f"Processing unit num {unum}/{total}")
        multi_raster_plot(
            sess,
            [
                ("Rp_Extra", sess.probe_trial_idxs),
                ("Rs", sess.saccade_trial_idxs),
                ("Rmixed", sess.mixed_trial_idxs)
            ],
            activity_idxs,
            unit_number=unum,
            passing_func=passing_func
        )


def main():
    matplotlib.use('Agg')   # Suppress matplotlib window opening

    filename = "2023-05-15_mlati7_updated_output"
    import time
    time.sleep(60*60)  # Wait an hour before starting..
    standard_all_summary(filename)
    tw = 2

    # sess = NWBSessionProcessor("../scripts", filename, "../graphs")

    # activity_idxs = sess.activity_filtered_units_idxs(sess.probe_zeta_idxs())
    # activity_idxs = sess.unfiltered_idxs()
    # activity_idxs = sess.probe_zeta_idxs()

    # probe_units, saccade_units, mixed_units, rp_peri_units = sess.filter_units(activity_idxs)

    # units_baseline_firingrate(mixed_units)
    # units_baseline_firingrate(probe_units)
    # units_baseline_firingrate(saccade_units)

    # mean_response(sess, "Rs", activity_idxs, sess.saccade_trial_idxs)
    # mean_response(sess, "Rp_Extra", activity_idxs, sess.probe_trial_idxs)
    # mean_response(sess, "Rmixed", activity_idxs, sess.mixed_trial_idxs)
    # mean_response_custom(np.mean(sess.rp_peri_units()[activity_idxs, :, :], axis=1), "Rp_Peri")

    # avg_raster_plot(sess, "Rs", activity_idxs, sess.saccade_trial_idxs, 1000)
    # avg_raster_plot(sess, "Rmixed", activity_idxs, sess.mixed_trial_idxs, 1000)
    # avg_raster_plot(sess, "Rp_Extra", activity_idxs, sess.probe_trial_idxs, 1000)

    # matplotlib.use('Agg')   # Suppress matplotlib window opening
    # for unum in range(len(activity_idxs)):

    # Graph individual units
    # passing_func = functools.partial(
    #     sess.passing_multi,
    #     [
    #         sess.passing_zeta,  # zeta test
    #         sess.passing_activity,  # activity test
    #     ]
    # )

    # total = len(activity_idxs)
    # # for unum in range(1):
    # for unum in range(total):
    #     print(f"Processing unit num {unum}/{total}")
    #     multi_raster_plot(
    #         sess,
    #         [
    #             ("Rp_Extra", sess.probe_trial_idxs),
    #             ("Rs", sess.saccade_trial_idxs),
    #             ("Rmixed", sess.mixed_trial_idxs)
    #         ],
    #         activity_idxs,
    #         unit_number=unum,
    #         passing_func=passing_func
    #     )

    # tot = len(activity_idxs)
    # for u in range(tot):
    #     print(f"{u}/{tot}")
    #     single_raster_plot(sess, "Rp_Extra", activity_idxs, sess.probe_trial_idxs, u)
    #     single_raster_plot(sess, "Rs", activity_idxs, sess.saccade_trial_idxs, u)
    #     single_raster_plot(sess, "Rmixed", activity_idxs, sess.mixed_trial_idxs, u)
    #
    tw = 2


if __name__ == "__main__":
    main()
