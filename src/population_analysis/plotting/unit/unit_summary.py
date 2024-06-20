from typing import Optional

import numpy as np

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.processors.nwb import NWBSession, UnitFilter
from population_analysis.processors.nwb.filters import BasicFilter, Filter
import matplotlib.pyplot as plt

from population_analysis.processors.nwb.filters.trial_filters.rp_peri import RelativeTrialFilter


def get_spike_idxs(bbool_counts, unit_number, trial_idxs, unit_filter: Optional[UnitFilter] = None):
    counts = bbool_counts

    if unit_filter is not None:
        counts = counts[unit_filter.idxs(), :, :]
    counts = counts[:, trial_idxs, :][unit_number]

    spike_idxss = []
    for trial in counts:
        spike_idxss.append(np.where(trial)[0])
    return spike_idxss


def unit_summary(sess: NWBSession, unit_num: int):
    """
    Graph a single unit's mean response of each trial type in each motion direction with rasters
    """

    # First we want to start with mean responses
    print("Rendering mean responses")
    motion_trial_filters = [
        ("motion=-1", sess.trial_motion_filter(-1)),
        ("motion=1", sess.trial_motion_filter(1)),
    ]

    response_trial_filters = [
        ("RpExtra", BasicFilter(sess.probe_trial_idxs, sess.num_trials)),
        ("Rs", BasicFilter(sess.saccade_trial_idxs, sess.num_trials)),
        ("Rmixed", BasicFilter(sess.mixed_trial_idxs, sess.num_trials))
    ]
    # Rows + 1 for rasters, Cols + 1 for rp peri
    fig, axs = plt.subplots(len(motion_trial_filters) + 1, len(response_trial_filters) + 1,
                            figsize=(16, 8))
    fig.subplots_adjust(wspace=0.2, hspace=.3)

    row_maxs = [0] * (len(response_trial_filters) + 1)
    row_mins = [999] * (len(response_trial_filters) + 1)
    xvals = np.array(range(NUM_FIRINGRATE_SAMPLES)) - 10  # 10 is time of probe

    row_idx = 0
    for mot_name, mot_filt in motion_trial_filters:
        col_idx = 0
        for resp_name, resp_filt in response_trial_filters:
            trial_filt = mot_filt.copy().append(resp_filt.copy())
            unitdata = sess.units()[:, trial_filt.idxs()][unit_num]
            avg = np.mean(unitdata, axis=0)  # Average over trials

            if row_maxs[row_idx] < np.max(avg):
                row_maxs[row_idx] = np.max(avg)
            if row_mins[row_idx] > np.min(avg):
                row_mins[row_idx] = np.min(avg)

            ax = axs[row_idx, col_idx]
            ax.plot(xvals, avg)
            col_idx = col_idx + 1
        row_idx = row_idx + 1

    # RpPeri
    rp_peri_units = sess.rp_peri_units()

    for i, mot in enumerate(motion_trial_filters):
        mot_name, mot_filt = mot
        filt = RelativeTrialFilter(mot_filt, sess.mixed_trial_idxs)
        ax = axs[i, len(response_trial_filters)]
        units = rp_peri_units[:, filt.idxs()][unit_num]
        avg = np.mean(units, axis=0)

        if row_maxs[i] < np.max(avg):
            row_maxs[i] = np.max(avg)
        if row_mins[i] > np.min(avg):
            row_mins[i] = np.min(avg)
        ax.plot(xvals, avg)
        ax.set_ylim(row_mins[i], row_maxs[i])
        ax.set_yticks([])

        if i == 0:
            ax.set_title("RpPeri")
        if i == len(motion_trial_filters) - 1:
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
    raster_axs = axs[len(motion_trial_filters)]
    spikes = sess.nwb.units["trial_spike_flags"]

    for idx, trial in enumerate(response_trial_filters):
        trial_name, trial_filt = trial
        ax = raster_axs[idx]
        spike_idxs = get_spike_idxs(spikes, unit_num, trial_filt.idxs())
        ax.eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)

    raster_axs[0].set_ylabel("Trial #")
    raster_axs[0].set_xlabel("Time (ms)")

    plt.savefig(f"unit-{unit_num}.png")


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSession("../../../../scripts", filename, "../../../../graphs")
    unit_filter = sess.unit_filter_qm().append(
        sess.unit_filter_probe_zeta().append(
            sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
        )
    )

    # for unit_num in unit_filter.idxs():
    # for unit_num in [0, 5]:
    for unit_num in Filter.empty(sess.units().shape[0]).idxs():
        print(f"Rendering unit {unit_num}..")
        unit_summary(sess, unit_num)


if __name__ == "__main__":
    main()

