import h5py
import numpy as np

from population_analysis.processors.experiments.saccadic_modulation.firing_rates import FiringRateCalculator
from population_analysis.processors.kilosort import KilosortProcessor
from population_analysis.processors.experiments.saccadic_modulation import SaccadicModulationTrialProcessor


def _calc_motions(timestamps, windows, motions, name):
    # Timestamps is [ts1, ts2, ..] (n,)
    # windows is [[start, end], ..] (grating_len, 2)
    # motions is [+-1, ..] (grating_len,)
    # returns [+-1, ..] (n,)
    if windows.shape[0] != motions.shape[0]:
        raise ValueError(f"Windows shape {windows.shape} != Motions shape {motions.shape} !")

    calcd = []
    passing_idxs = []
    for tidx, timestamp in enumerate(timestamps):
        for widx, window in enumerate(windows):
            w_start, w_end = window
            if w_start <= timestamp <= w_end:
                calcd.append(motions[widx])
                passing_idxs.append(tidx)
                break

    print(f"Found {len(timestamps) - len(calcd)} timestamps during static grating for {name}, excluding..")
    calcd = np.array(calcd)
    # filter out timestamps during static grating by not including ones that aren't in the window by passing index
    passing_idxs = np.array(passing_idxs)
    filtered_timestamps = timestamps[passing_idxs]

    return filtered_timestamps, calcd, passing_idxs


def _event_timings(raw_data, grating_windows):
    # Cutoff timing is to ignore any events before given time
    cutoff_time = grating_windows["grating_timestamps"][0][0]

    # Probe and Saccade timestamps
    probe_timestamps = np.array(raw_data["stimuli"]["dg"]["probe"]["timestamps"])


    # Saccades from the left eye TODO other eyes?
    saccade_timestamps = np.array(raw_data["saccades"]["predicted"]["left"]["timestamps"])
    saccade_directions = np.array(raw_data["saccades"]["predicted"]["left"]["labels"])
    direction_idxs = np.where(saccade_directions != 0)[0]

    saccade_timestamps = saccade_timestamps[direction_idxs][:, 0]  # Use the start of the saccade time window as 'saccade event time'

    # Cut off any probes/saccades that occur before our cutoff window
    probe_timestamps = probe_timestamps[np.where(probe_timestamps >= cutoff_time)[0]]
    saccade_timestamps = saccade_timestamps[np.where(saccade_timestamps >= cutoff_time)[0]]

    grating_motion_directions = np.array(raw_data["stimuli"]["dg"]["grating"]["motion"])
    grating_windows = grating_windows["grating_timestamps"]

    probe_timestamps, probe_motions, probe_blocks = _calc_motions(probe_timestamps, grating_windows, grating_motion_directions, "probes")
    saccade_timestamps, saccade_motions, saccade_blocks = _calc_motions(saccade_timestamps, grating_windows, grating_motion_directions, "saccades")

    return {
        "saccade_timestamps": saccade_timestamps,
        "saccade_motions": saccade_motions,
        "saccade_blocks": saccade_blocks,

        "probe_timestamps": probe_timestamps,
        "probe_motions": probe_motions,
        "probe_blocks": probe_blocks,
        "grating_motion_direction": grating_motion_directions,
        "grating_windows": grating_windows
    }


def _calc_grating_windows(raw_data):
    # zip up timestamps of the drifting grating with the corresponding iti (inter time intervals) to give [[grating_start, grating_stop], ..]
    window_timestamps = np.array(list(zip(list(raw_data["stimuli"]["dg"]["grating"]["timestamps"]), list(raw_data["stimuli"]["dg"]["iti"]["timestamps"]))))
    inter_grating_timestamps = []
    for i in range(1, len(window_timestamps)):
        _, last_stop = window_timestamps[i - 1]
        next_start, _ = window_timestamps[i]
        inter_grating_timestamps.append([last_stop, next_start])

    inter_grating_timestamps = np.array(inter_grating_timestamps)  # Timestamps of no motion of the drifting grating in [[dg_stop, dg_start], ..]

    return {
        "grating_timestamps": window_timestamps,
        "inter_grating_timestamps": inter_grating_timestamps
    }


def main():
    raw_data = h5py.File("output.hdf")
    spike_clusters = np.array(raw_data["spikes"]["clusters"])
    spike_timings = np.array(raw_data["spikes"]["timestamps"])

    kp = KilosortProcessor(spike_clusters, spike_timings)

    fr, fr_bins = kp.calculate_firingrates(20, True)  # Bin size is 20ms in seconds
    # sp = kp.calculate_spikes(True)
    import matplotlib.pyplot as plt
    grating_windows = _calc_grating_windows(raw_data)
    # Grab the trials for the events
    events = _event_timings(raw_data, grating_windows)

    # Separate the trials into Rs, RpExtra and Rmixed
    smp = SaccadicModulationTrialProcessor(fr_bins, events)
    trials = smp.calculate()

    tfrc = FiringRateCalculator(fr, trials)
    firing_rates = tfrc.calculate()

    tw = 2


if __name__ == "__main__":
    main()
