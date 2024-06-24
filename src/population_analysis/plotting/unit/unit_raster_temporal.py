import os.path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from population_analysis.processors.nwb import NWBSession
from population_analysis.processors.nwb.filters import BasicFilter, Filter


def do_raster(sess, unit_filter):
    trial_start_stops = sess.trial_durations()
    trials_spikes = sess.spikes()
    units = sess.units()

    offsets = {
        "probe": 0,
        "saccade": 700,
        "mixed": 700 * 2
    }

    def do_offset(val, name):
        off = offsets[name]
        newv = []
        for v in val:
            newv.append(v + off)
        return newv

    trial_types = {
        "probe": sess.probe_trial_idxs,
        "saccade": sess.saccade_trial_idxs,
        "mixed": sess.mixed_trial_idxs
    }

    all_trial_durations = []  # [(name, [start, stop], trial_units_spikes, trial_idx, trial_units_waveforms]
    for trial_type, trial_idxs in trial_types.items():
        print(f"Processing {trial_type}..")
        for trial_idx in trial_idxs:
            start_stop = trial_start_stops[trial_idx]
            # waveforms = units[:, trial_idx, :]
            spikes = trials_spikes[:, trial_idx, :]
            all_trial_durations.append([
                trial_type,
                start_stop,
                spikes,
                trial_idx,
                # waveforms,
            ])
    print("Sorting..")
    sorted_durations = sorted(all_trial_durations, key=lambda x: x[1][0])  # Sort by start time

    for unit_num in unit_filter.idxs():
        print(f"Rendering unit {unit_num}")
        eventplot_data = []
        offset_eventplot_data = []

        for trial in sorted_durations:
            spike_idxs = trial[2][unit_num]  # trial_units_spikes
            spike_idxs = np.where(spike_idxs)[0]
            spike_idxs_offset = do_offset(spike_idxs, trial[0])  # trial type
            eventplot_data.append(spike_idxs)
            offset_eventplot_data.append(spike_idxs_offset)
            tw = 2

        fig, axs = plt.subplots(nrows=1, ncols=2)

        axs[0].eventplot(offset_eventplot_data, colors="black", lineoffsets=1, linelengths=1)
        axs[0].set_xlabel("Probes,   Saccades,    Mixed")

        axs[1].eventplot(eventplot_data, colors="black", lineoffsets=1, linelengths=1)

        axs[1].set_xlabel("All trials aligned")

        axs[0].hlines(len(offset_eventplot_data), 0, 2100)
        axs[0].hlines(0, 0, 2100)

        axs[1].hlines(len(eventplot_data), 0, 700)
        axs[1].hlines(0, 0, 700)
        fig.suptitle(f"Unit {unit_num}")

        fig.savefig(f"unit_raster_temporal_plots/time_raster_u{unit_num}.png")
        plt.close(fig)
        tw = 2

    tw = 2


def main():
    # filename = "not_smoothed_2023-05-15_mlati7_output"
    filename = "2023-05-15_mlati7_output"

    matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSession("../../../../scripts", filename, "../graphs")

    if not os.path.exists("unit_raster_temporal_plots"):
        os.mkdir("unit_raster_temporal_plots")

    # do_raster(sess, BasicFilter([233], sess.num_units))
    do_raster(sess, Filter.empty(sess.num_units))

    pass


if __name__ == "__main__":
    main()

