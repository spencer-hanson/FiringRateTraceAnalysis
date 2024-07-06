import math
import os.path
from multiprocessing import Pool
from typing import Optional

import numpy as np

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.processors.filters import BasicFilter, Filter
from population_analysis.processors.filters.unit_filters import UnitFilter
from population_analysis.sessions.saccadic_modulation import NWBSession
import matplotlib.pyplot as plt


def get_spike_idxs(bbool_counts, unit_number, trial_idxs, unit_filter: Optional[UnitFilter] = None):
    counts = bbool_counts

    if unit_filter is not None:
        counts = counts[unit_filter.idxs(), :, :]
    counts = counts[:, trial_idxs, :][unit_number]

    spike_idxss = []
    for trial in counts:
        spike_idxss.append(np.where(trial)[0])
    return spike_idxss


def unit_summary(sess: NWBSession, unit_num: int, foldername: str):
    """
    Graph a single unit's mean response of each trial type in each motion direction with rasters
    """

    # First we want to start with mean responses
    print("Rendering mean responses")
    motion_trial_filters = [
        ("motion=-1", -1),
        ("motion=1", 1),
    ]

    response_trial_filters = [
        ("RpExtra", (BasicFilter, sess.probe_trial_idxs, sess.num_trials)),
        ("Rs", (BasicFilter, sess.saccade_trial_idxs, sess.num_trials)),
        ("Rmixed", (BasicFilter, sess.mixed_trial_idxs, sess.num_trials))
    ]
    # Rows + 2 for rasters, Cols + 1 for rp peri
    fig, axs = plt.subplots(len(motion_trial_filters) + 2, len(response_trial_filters) + 1,
                            figsize=(16, 8))
    fig.subplots_adjust(wspace=0.2, hspace=.3)

    row_maxs = [-9999] * len(motion_trial_filters)
    row_mins = [9999] * len(motion_trial_filters)
    xvals = np.array(range(NUM_FIRINGRATE_SAMPLES)) - 10  # 10 is time of probe

    row_idx = 0
    for mot_name, motdir in motion_trial_filters:
        col_idx = 0
        for resp_name, resp_filt in response_trial_filters:
            tfilt = resp_filt[0](*resp_filt[1:])
            trial_filt = sess.trial_motion_filter(motdir).append(tfilt)
            unitdata = sess.units()[:, trial_filt.idxs()][unit_num]
            avg = np.mean(unitdata, axis=0)  # Average over trials

            if np.max(avg) > row_maxs[row_idx]:
                row_maxs[row_idx] = np.max(avg)

            if np.min(avg) < row_mins[row_idx]:
                row_mins[row_idx] = np.min(avg)

            ax = axs[row_idx, col_idx]
            ax.plot(xvals, avg)
            col_idx = col_idx + 1
        row_idx = row_idx + 1

    # RpPeri
    rp_peri_units = sess.rp_peri_units()
    for rpp_idx, mot in enumerate(motion_trial_filters):
        mot_name, motdir = mot
        mot_filt = sess.trial_motion_filter(motdir)
        filt = sess.trial_filter_rp_peri(mot_filt)
        ax = axs[rpp_idx, len(response_trial_filters)]
        units = rp_peri_units[:, filt.idxs()][unit_num]
        avg = np.mean(units, axis=0)

        if np.max(avg) > row_maxs[rpp_idx]:
            row_maxs[rpp_idx] = np.max(avg)

        if np.min(avg) < row_mins[rpp_idx]:
            row_mins[rpp_idx] = np.min(avg)

        ax.plot(xvals, avg)
        ax.set_ylim(row_mins[rpp_idx], row_maxs[rpp_idx])
        ax.set_yticks([])

        if rpp_idx == 0:
            ax.set_title("RpPeri")
        if rpp_idx == len(motion_trial_filters) - 1:
            ax.set_xlabel("Time from probe (20ms bins)")
        else:
            ax.set_xticks([])

    # Format plots
    for row_idx in range(len(motion_trial_filters)):
        for col_idx in range(len(response_trial_filters)):
            ax = axs[row_idx, col_idx]
            ax.set_ylim(row_mins[row_idx], row_maxs[row_idx])
            if row_idx == 0:
                ax.set_title(response_trial_filters[col_idx][0])
            if col_idx != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(motion_trial_filters[row_idx][0])

            if row_idx != len(motion_trial_filters)-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time from probe (20ms bins)")

    # Raster plots
    print("Rendering rasters")
    for motidx, motdir in enumerate([-1, 1]):
        raster_axs = axs[len(motion_trial_filters) + motidx]
        spikes = sess.spikes()

        for idx, trial in enumerate(response_trial_filters):
            trial_name, trialdata = trial
            trial_filt = trialdata[0](*trialdata[1:])
            trial_filt = trial_filt.append(sess.trial_motion_filter(motdir))

            ax = raster_axs[idx]
            spike_idxs = get_spike_idxs(spikes, unit_num, trial_filt.idxs())
            ax.eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)
            ax.set_title(f"{trial_name} motion={motdir}")
            ax.set_xlim([0, 700])
            ax.set_xticks(np.arange(0, 700, 100))

        raster_axs[0].set_ylabel("Trial #")
        raster_axs[0].set_xlabel("Time (ms)")

    # plt.show()
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    plt.savefig(f"{foldername}/unit-{unit_num}.png")
    plt.close(fig)
    tw = 2


def _calc_pool_args(num_pools, sample_list, sess):
    num_samples = len(sample_list)

    val = math.floor(num_samples / num_pools)
    remainder = num_samples % num_pools

    nums = [val] * num_pools
    nums[-1] = (nums[-1] + remainder, True)
    # return nums

    # [filepath, filename], [2, .. 28 etc], False unless remainder pool, foldername is sess name
    # sess_args, sample_list, display, foldername


def multiprocess_func(sess_args, sample_list, display, foldername):
    sess = NWBSession(*sess_args)
    num_samples = len(sample_list)

    if display:
        print("Rendering units with multiprocessing -", end="")

    one_tenth = int(num_samples / 10)
    one_tenth = one_tenth if one_tenth != 0 else 1

    # start = pendulum.now()
    for progress, unit_num in enumerate(sample_list):

        if display and progress % one_tenth == 0:
            print(f" {round(progress / num_samples, 2)*100}%", end="")

        unit_summary(sess, unit_num, foldername)


def main():
    # filepath = "../../../../scripts"
    # filename = "new_test"
    # filename = "output-mlati6-2023-05-12.hdf-nwb"
    filepath = "../../../../scripts/generated"
    filename = "generated.hdf-nwb"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    sess_args = [filepath, filename]
    sess = NWBSession(*sess_args)

    ufilt = BasicFilter.empty(sess.num_units)

    num_pools = 8
    with Pool(num_pools + 1) as p:
        p.map(
            multiprocess_func,
            _calc_pool_args(num_pools, ufilt.idxs(), sess)
        )  # TODO Finish argument mapping for pool multiprocessing

    # for unit_num in Filter.empty(sess.num_units).idxs():
    # for unit_num in range(sess.num_units-1, 0, -1):
    # for unit_num in range(320, 0, -1):
    # for unit_num in range(30, sess.num_units):
    # for unit_num in [107?, 206?]:
    #     print(f"Rendering unit {unit_num}..")
    #     unit_summary(sess, unit_num, "default_unit_summary")


if __name__ == "__main__":
    main()

