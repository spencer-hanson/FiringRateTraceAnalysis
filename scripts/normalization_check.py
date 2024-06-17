import numpy as np

from population_analysis.processors.nwb import NWBSessionProcessor
from population_analysis.processors.nwb.unit_normalization import UnitNormalizer
from population_analysis.processors.raw import RawSessionProcessor


def main():
    raw_sess = RawSessionProcessor("output.hdf", "mlati7")
    sess = NWBSessionProcessor("../scripts", "2023-05-15_mlati7_output", "../graphs")

    trial_event_idxs = sess.nwb.processing["behavior"]["trial_event_idxs"].data[:]  # trial_event in idxs
    trial_duration_idxs = sess.nwb.processing["behavior"]["trial_durations_idxs"].data[:]  # [trial_start, trial_stop] in idxs
    trial_event_duration_idxs = []
    for ev_idx in range(len(trial_event_idxs)):
        trial_event_duration_idxs.append(
            [trial_duration_idxs[ev_idx][0], trial_event_idxs[ev_idx], trial_duration_idxs[ev_idx][1]]  # start, event, stop
        )

    trial_event_duration_idxs = np.array(trial_event_duration_idxs) + raw_sess._raw_spike_idx_offset  # Add offset for truncated experiment fix

    norm = UnitNormalizer(
        raw_sess._raw_spike_clusters,
        raw_sess._raw_spike_timestamps,
        trial_event_duration_idxs,
        unit_nums  # want the cluster specific unit numbers to iterate over
    )

    norm.normalize()
    tw = 2


if __name__ == "__main__":
    main()
