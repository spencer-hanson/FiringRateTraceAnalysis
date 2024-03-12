from typing import Optional

import h5py
import numpy as np
import pendulum
from pynwb import TimeSeries
from pynwb.file import Subject
from simply_nwb import SimpleNWB

from population_analysis.consts import PRE_TRIAL_MS, POST_TRIAL_MS, SESSION_DESCRIPTION, EXPERIMENTERS, \
    EXPERIMENT_DESCRIPTION, MOUSE_DETAILS, EXPERIMENT_KEYWORDS, DEVICE_NAME, DEVICE_DESCRIPTION, DEVICE_MANUFACTURER
from population_analysis.population.units import UnitPopulation


class RawSessionProcessor(object):
    def __init__(self, filename):
        data = h5py.File(filename)
        self.spike_clusters = np.array(data["spikes"]["clusters"])
        self.spike_timestamps = np.array(data["spikes"]["timestamps"])
        self.probe_timestamps = np.array(data["stimuli"]["dg"]["probe"]["timestamps"])
        self.saccade_timestamps = np.array(data["saccades"]["predicted"]["left"]["nasal"]["timestamps"])
        self._unit_pop: Optional[UnitPopulation] = None

    def calc_unit_population_stats(self):
        unit_pop = UnitPopulation(self.spike_timestamps, self.spike_clusters)

        print("Extracting saccade spike timestamps..")
        saccade_spike_range_idxs = self._extract_timestamp_idxs(self.spike_timestamps, self.saccade_timestamps)
        print("Extracting probe spike timestamps..")
        probe_spike_range_idxs = self._extract_timestamp_idxs(self.spike_timestamps, self.probe_timestamps)

        trials = self._demix_trials(saccade_spike_range_idxs, probe_spike_range_idxs)

        unit_pop.add_probe_trials(trials["probe"])
        unit_pop.add_saccade_trials(trials["saccade"])
        unit_pop.add_mixed_trials(trials["mixed"])

        unit_pop.calc_firingrates()

        self._unit_pop = unit_pop

    def save_to_nwb(self, filename, mouse_name, session_id):
        print(f"Saving to NWB file '{filename}'")

        if self._unit_pop is None:
            self.calc_unit_population_stats()

        birthday_diff = pendulum.now().diff(MOUSE_DETAILS[mouse_name]["birthday"])

        nwb = SimpleNWB.create_nwb(
            # Required
            session_description=SESSION_DESCRIPTION,
            # Subtract 1 year so we don't run into the 'NWB start time is at a greater date than current' issue
            session_start_time=pendulum.now().subtract(years=1),
            experimenter=EXPERIMENTERS,
            lab="Felsen Lab",
            experiment_description=EXPERIMENT_DESCRIPTION,
            # Optional
            identifier=mouse_name,
            subject=Subject(**{
                "subject_id": mouse_name,
                "age": f"P{birthday_diff.days}D",  # ISO-8601 for days duration
                "strain": MOUSE_DETAILS[mouse_name]["strain"],
                "description": f"Mouse id '{mouse_name}'",
                "sex": MOUSE_DETAILS[mouse_name]["sex"]
            }),
            session_id=session_id,
            institution="CU Anschutz",
            keywords=EXPERIMENT_KEYWORDS,
            # related_publications="DOI::LINK GOES HERE FOR RELATED PUBLICATIONS"
        )

        # Add device
        nwb.create_device(
            name=DEVICE_NAME, description=DEVICE_DESCRIPTION, manufacturer=DEVICE_MANUFACTURER
        )

        # Add units
        nwb.add_unit_column(name="trial_firing_rates",
                            description="trials x waveform length array for each unit's presence in a trial")
        for unit_num in range(self.unit_pop.num_units):
            unit_spike_idxs = np.where(self.unit_pop.spike_clusters == unit_num)
            spike_times = self.unit_pop.spike_timestamps[unit_spike_idxs]
            nwb.add_unit(
                spike_times=spike_times,
                trial_firing_rates=self.unit_pop.unit_firingrates[unit_num]
            )

        # Add probe and saccade event timings, trial types
        behavior_events = nwb.create_processing_module(name="behavior",
                                                       description="Contains saccade and probe event timings")

        probe_ts = TimeSeries(name="probes", data=self.probe_timestamps, unit="s", rate=0.001)
        saccade_ts = TimeSeries(name="saccades", data=self.saccade_timestamps, unit="s", rate=0.001)

        trial_types = np.array(self.unit_pop.get_trial_labels())
        unique_trial_types = np.unique(trial_types)
        for trial_type in unique_trial_types:
            behavior_events.add(TimeSeries(name=f"trial-{trial_type}",
                                           data=np.where(trial_types == trial_type)[0], rate=1.0, unit="idx"))
        behavior_events.add(probe_ts)
        behavior_events.add(saccade_ts)
        print("Writing to file, may take a while..")
        SimpleNWB.write(nwb, filename)
        print("Done!")

    @property
    def unit_pop(self):
        if self._unit_pop is None:
            self.calc_unit_population_stats()
        return self._unit_pop

    def _remove_by_idxs(self, lst: list, idxs: list[int]) -> list:
        # Filter out a list by indexes to not include
        l2 = []
        for idx, l in enumerate(lst):
            if idx in idxs:
                continue
            else:
                l2.append(l)
        return l2

    def _extract_timestamp_idxs(self, spike_timestamps, other_timestamps) -> list[[float, float, float]]:
        # return indices into spike_timestamps within a window of -200ms to +700ms for trials times in other_timestamps
        idx_ranges = []

        other_len = len(other_timestamps)
        other_one_tenth = int(1 / 10 * other_len)
        for idx, ts in enumerate(other_timestamps):
            if idx % other_one_tenth == 0:
                print(f" {round(100 * (idx / other_len), 2)}%", end="")
            if np.isnan(ts):
                # idx_ranges.append(None)
                continue
            start_idx = np.where(ts - (PRE_TRIAL_MS / 1000) < spike_timestamps)[0][
                0]  # First index in tuple, first index is the edge
            end_idx = np.where(ts + (POST_TRIAL_MS / 1000) <= spike_timestamps)[0][0]
            ts_idx = np.where(ts >= spike_timestamps)[0][-1]  # Index of the event timestamp itself, next smallest value
            idx_ranges.append([start_idx, ts_idx, end_idx])
        print("")
        return idx_ranges

    def _demix_trials(self, saccade_idxs: list[list[float]], probe_idxs: list[list[float]]):
        # If a saccade and probe trial occur within +- .5 sec (500ms) then they should be considered a mixed trial
        trials = {
            "saccade": [],
            "probe": [],
            "mixed": []
        }

        # Find probes that occur within 500ms of a saccade, remove them from the list of possible
        # saccades/probes after finding them
        def within_window(list1, list2, label1, label2):  # Find events within eachothers bounds
            mixed = []
            l1_to_remove = []
            l2_to_remove = []

            for f_idx, first_idx in enumerate(list1):
                f_start = first_idx[0]  # first start
                f_end = first_idx[2]

                for s_idx, second_idx in enumerate(list2):
                    s_event = second_idx[1]  # second event time idx
                    if f_start <= s_event <= f_end:  # Found mixed
                        mixed.append({
                            label1: first_idx,
                            label2: second_idx
                        })
                        l2_to_remove.append(s_idx)
                        l1_to_remove.append(f_idx)
                        break
                list2 = self._remove_by_idxs(list2, l2_to_remove)
                l2_to_remove = []
            list1 = self._remove_by_idxs(list1, l1_to_remove)
            return [list1, list2, mixed]

        # Find saccades that occur within 500ms of a probe
        print("Demixing saccades within 500ms from a probe")
        saccade_idxs, probe_idxs, both = within_window(saccade_idxs, probe_idxs, "saccade", "probe")

        # Find probes that occur within 500ms of a saccade
        print("Demixing probes within 500ms of a saccade")
        probe_idxs, saccade_idxs, both2 = within_window(probe_idxs, saccade_idxs, "probe", "saccade")

        trials["saccade"] = saccade_idxs
        trials["probe"] = probe_idxs
        trials["mixed"] = [*both, *both2]

        return trials
