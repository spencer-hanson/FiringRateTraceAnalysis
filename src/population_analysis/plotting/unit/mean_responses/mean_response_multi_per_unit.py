import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import get_xaxis_vals
from population_analysis.processors.filters import BasicFilter
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_rp_peri_unit(sess, ax, motdir, unit_num, row_min, row_max):
    rp_peri_units = sess.rp_peri_units()
    mot_filt = sess.trial_motion_filter(motdir)
    filt = sess.trial_filter_rp_peri(mot_filt)

    units = rp_peri_units[:, filt.idxs()][unit_num]
    avg = np.mean(units, axis=0)

    if np.max(avg) > row_max:
        row_max = np.max(avg)

    if np.min(avg) < row_min:
        row_min = np.min(avg)

    ax.plot(get_xaxis_vals(), avg)
    ax.set_ylim(row_min, row_max)

    return row_min, row_max


def plot_rp_extra_unit(sess, ax, motdir, unit_num, row_min, row_max):
    filt = sess.trial_motion_filter(motdir).append(BasicFilter(sess.probe_trial_idxs, sess.num_trials))
    return plot_unit(sess, ax, filt, unit_num, row_min, row_max)


def plot_rs_unit(sess, ax, motdir, unit_num, row_min, row_max):
    filt = sess.trial_motion_filter(motdir).append(BasicFilter(sess.saccade_trial_idxs, sess.num_trials))
    return plot_unit(sess, ax, filt, unit_num, row_min, row_max)


def plot_r_mixed_unit(sess, ax, motdir, unit_num, row_min, row_max):
    filt = sess.trial_motion_filter(motdir).append(BasicFilter(sess.mixed_trial_idxs, sess.num_trials))
    return plot_unit(sess, ax, filt, unit_num, row_min, row_max)


def plot_unit(sess, ax, filt, unit_num, row_min, row_max):
    units = sess.units()[:, filt.idxs()][unit_num]
    avg = np.mean(units, axis=0)

    if np.max(avg) > row_max:
        row_max = np.max(avg)

    if np.min(avg) < row_min:
        row_min = np.min(avg)

    ax.plot(get_xaxis_vals(), avg)
    ax.set_ylim(row_min, row_max)

    return row_min, row_max


def plot_multi_mean_responses(sess, unit_filter, filename):

    datas = [
        ("RpExtra", plot_rp_extra_unit),
        ("Rs", plot_rs_unit),
        ("Rmixed", plot_r_mixed_unit),
        ("RpPeri", plot_rp_peri_unit),
    ]
    save_foldername = f"units-{filename}"

    if not os.path.exists(save_foldername):
        os.mkdir(save_foldername)

    for unit_num in unit_filter.idxs():
        fullsave = os.path.join(save_foldername, f"unit-{unit_num}.png")
        if os.path.exists(fullsave):
            print(f"Graph '{fullsave}' exists, skipping..")
            continue

        fig, axs = plt.subplots(2, len(datas), figsize=(16, 6), sharey=True)
        fig.subplots_adjust(wspace=0.2, hspace=.3)

        row_mins = [999, 999]  # 2 len for each motion dir
        row_maxs = [-999, -999]

        for idx, funcdata in enumerate(datas):
            name, func = funcdata

            row_min = row_mins[0]
            row_max = row_maxs[0]
            row_min, row_max = func(sess, axs[0, idx], -1, unit_num, row_min, row_max)
            row_mins[0] = row_min
            row_maxs[0] = row_max

            row_min = row_mins[1]
            row_max = row_maxs[1]
            row_min, row_max = func(sess, axs[1, idx], 1, unit_num, row_min, row_max)
            row_mins[1] = row_min
            row_maxs[1] = row_max

            if idx == 0:
                axs[0, 0].set_ylabel("motion=-1")
                axs[1, 0].set_ylabel("motion=1")
            axs[0, idx].set_title(name)

        print(f"Saving '{fullsave}'..")
        plt.savefig(fullsave)
        plt.close(fig)


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    nwbfiles = glob.glob("../../../../../scripts/*/*.nwb")

    for nwb_filename in nwbfiles:
        filepath = os.path.dirname(nwb_filename)
        filename = os.path.basename(nwb_filename)[:-len(".nwb")]
        print(f"Processing {nwb_filename}..")

        sess = NWBSession(filepath, filename)
        unit_filter = BasicFilter.empty(sess.num_units)

        plot_multi_mean_responses(sess, unit_filter, filename)


if __name__ == "__main__":
    main()
