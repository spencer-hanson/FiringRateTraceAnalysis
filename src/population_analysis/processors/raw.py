from typing import Optional

import h5py
import numpy as np
import pendulum
from pynwb import TimeSeries
from pynwb.file import Subject
from simply_nwb import SimpleNWB

from population_analysis.consts import PRE_TRIAL_MS, POST_TRIAL_MS, SESSION_DESCRIPTION, EXPERIMENTERS, \
    EXPERIMENT_DESCRIPTION, MOUSE_DETAILS, EXPERIMENT_KEYWORDS, DEVICE_NAME, DEVICE_DESCRIPTION, DEVICE_MANUFACTURER, \
    NUM_BASELINE_POINTS, UNIT_ZETA_P_VALUE, TOTAL_TRIAL_MS, METRIC_NAMES, MIXED_THRESHOLD
from population_analysis.population.units import UnitPopulation


class RawSessionProcessor(object):
    def __init__(self, filename, mouse_name):
        data = h5py.File(filename)

        self._unit_pop: Optional[UnitPopulation] = None
        self._convert_unit_indexes = None

        # [[start, stop], ..] of each drifting grating. Any time outside these ranges is a non-moving static gray screen and shouldn't be included
        self.grating_timestamps = np.array(list(zip(list(data["stimuli"]["dg"]["grating"]["timestamps"]), list(data["stimuli"]["dg"]["iti"]["timestamps"]))))
        self.inter_grating_timestamps = self._calc_inter_grating_timestamps(self.grating_timestamps)
        self.motion_directions = np.array(data["stimuli"]["dg"]["grating"]["motion"])
        first_dg = self.grating_timestamps[0][0]

        # Spike Clusters and Timestamps
        self.spike_clusters = np.array(data["spikes"]["clusters"])
        self.spike_timestamps = np.array(data["spikes"]["timestamps"])
        # Find the indexes of the spike timestamps that are greater than the first drifting grating motion
        spike_ts_idxs = np.where(self.spike_timestamps >= first_dg)[0]
        self.spike_clusters = self.spike_clusters[spike_ts_idxs]
        self.spike_timestamps = self.spike_timestamps[spike_ts_idxs]
        self.pre_filtered_unique_units = np.unique(np.array(data["spikes"]["clusters"]))
        self.unique_units = np.unique(self.spike_clusters)

        # Probe and Saccade timestamps
        self.probe_timestamps = np.array(data["stimuli"]["dg"]["probe"]["timestamps"])
        self.probe_timestamps = self.probe_timestamps[np.where(self.probe_timestamps >= first_dg)[0]]
        self.saccade_timestamps = self._extract_saccade_timestamps(data["saccades"]["predicted"]["left"])  # Saccades from the left eye TODO other eyes?
        self.saccade_timestamps = self.saccade_timestamps[np.where(self.saccade_timestamps >= first_dg)[0]]

        self.probe_zeta = self._convert_unit_arr(np.array(data["zeta"]["probe"]["left"]["p"]))
        self.saccade_zeta = self._convert_unit_arr(np.array(data["zeta"]["saccade"]["nasal"]["p"]))

        self.metrics = self._extract_metrics(data)
        self.p_value_truth = self._calc_p_value_truth(self.probe_zeta, self.saccade_zeta)

        self.mouse_name = mouse_name
        self.mouse_birthday = MOUSE_DETAILS[mouse_name]["birthday"]
        self.mouse_strain = MOUSE_DETAILS[mouse_name]["strain"]
        self.mouse_sex = MOUSE_DETAILS[mouse_name]["sex"]

    def _convert_unit_arr(self, arr):
        # Convert a unit array from the original source unit ordering to the filtered version
        # For example, before filtering by drifting gratings, there are 562 units, but after there are only 552
        # Need to convert an array meant for the original 562 to the 552 version

        if self._convert_unit_indexes is None:
            indexes = []
            for unit_num in self.unique_units:
                # Index into pre-filtered units of where this unit number is
                pre_filtered_index = np.where(self.pre_filtered_unique_units == unit_num)[0][0]
                indexes.append(pre_filtered_index)
            self._convert_unit_indexes = indexes

        converted_array = arr[self._convert_unit_indexes]
        return converted_array

    def _calc_inter_grating_timestamps(self, grating_timestamps):
        inter = []
        for i in range(1, len(grating_timestamps)):
            _, last_stop = grating_timestamps[i - 1]
            next_start, _ = grating_timestamps[i]
            inter.append([last_stop, next_start])
        return inter

    def _extract_metrics(self, hd5data):
        metrics = {}
        for k, v in METRIC_NAMES.items():
            metrics[v] = self._convert_unit_arr(np.array(hd5data["metrics"][k]))
        return metrics

    def _extract_saccade_timestamps(self, saccade_data):
        direction = np.array(saccade_data["labels"])
        direction_idxs = np.where(direction != 0)[0]
        timestamps = np.array(saccade_data["timestamps"])[:, 0]   # Use the start of the saccade time window
        directional_timestamps = timestamps[direction_idxs]
        return directional_timestamps

    def _calc_p_value_truth(self, probe_zeta, saccade_zeta):
        # Calculate a bool array of if the units pass the p-value zeta test
        probe = probe_zeta <= UNIT_ZETA_P_VALUE
        saccade = saccade_zeta <= UNIT_ZETA_P_VALUE
        combined = np.logical_or(probe, saccade)  # A unit can pass probe or saccade to be included
        return combined

    def _filter_grating_windows(self, timestamp_event_idxs):
        # Filter out the trials that intersect with the drifting grating stopping (aren't within the window)
        # timestamp_event_idxs is [[start, time, end], ...] of the timings of the trial
        passing_trials = []
        for trial in timestamp_event_idxs:
            start_idx, event_idx, stop_idx = trial
            start_time = self.spike_timestamps[start_idx]
            stop_time = self.spike_timestamps[stop_idx]
            passes = True
            # The start and end of the inter-grating timestamps
            for inter_start, inter_end in self.inter_grating_timestamps:
                if inter_start < start_time < inter_end:  # Start time is within the inter window
                    passes = False
                    break
                elif inter_start < stop_time < inter_end:  # End time is within the inter window
                    passes = False
                    break

            if passes:
                passing_trials.append(trial)
        print(f"Filtered out {len(timestamp_event_idxs) - len(passing_trials)} trials that intersect with static stimuli")
        return passing_trials

    def _add_motion_direction(self, trials):
        # trials comes in as a dict like {'saccade': <data>, "probe": <data>, "mixed": {"saccade": <data>, "probe":<data>}
        # where <data> = [[start, event, stop],..] in idxs

        def calc_direction(start_idx):
            ts_start = self.spike_timestamps[start_idx]
            gratings_later_than = self.grating_timestamps[:, 0] < ts_start
            grating_motion_idx = np.where(gratings_later_than)[0][-1]  # Take the last entry, latest event
            direction = self.motion_directions[grating_motion_idx]
            return direction, grating_motion_idx

        for ky in ["saccade", "probe"]:
            for trial_idx in range(len(trials[ky])):
                start, _, _ = trials[ky][trial_idx]
                trials[ky][trial_idx].extend(calc_direction(start))

        for trial in trials["mixed"]:
            start, _, _ = trial["probe"]
            trial["probe"].extend(calc_direction(start))

        return trials

    def calc_unit_population_stats(self):
        unit_pop = UnitPopulation(self.spike_timestamps, self.spike_clusters, self.p_value_truth)

        print("Extracting saccade spike timestamps..")
        saccade_spike_range_idxs = self._extract_timestamp_idxs(self.spike_timestamps, self.saccade_timestamps)
        saccade_spike_range_idxs = self._filter_grating_windows(saccade_spike_range_idxs)

        print("Extracting probe spike timestamps..")
        probe_spike_range_idxs = self._extract_timestamp_idxs(self.spike_timestamps, self.probe_timestamps)
        probe_spike_range_idxs = self._filter_grating_windows(probe_spike_range_idxs)

        trials = self._demix_trials(saccade_spike_range_idxs, probe_spike_range_idxs)

        trials = self._add_motion_direction(trials)  # TODO also include which section matched trials

        unit_pop.add_probe_trials(trials["probe"])
        unit_pop.add_saccade_trials(trials["saccade"])
        unit_pop.add_mixed_trials(trials["mixed"])

        unit_pop.calc_firingrates()

        self._unit_pop = unit_pop

    def save_to_nwb(self, filename, session_id):
        if self._unit_pop is None:
            self.calc_unit_population_stats()

        birthday_diff = pendulum.now().diff(self.mouse_birthday)

        nwb = SimpleNWB.create_nwb(
            # Required
            session_description=SESSION_DESCRIPTION,
            # Subtract 1 year so we don't run into the 'NWB start time is at a greater date than current' issue
            session_start_time=pendulum.now().subtract(years=1),
            experimenter=EXPERIMENTERS,
            lab="Felsen Lab",
            experiment_description=EXPERIMENT_DESCRIPTION,
            # Optional
            identifier=self.mouse_name,
            subject=Subject(**{
                "subject_id": self.mouse_name,
                "age": f"P{birthday_diff.days}D",  # ISO-8601 for days duration
                "strain": self.mouse_strain,
                "description": f"Mouse id '{self.mouse_name}'",
                "sex": self.mouse_sex
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
        firing_rate_description = f"trials x waveform length array for each unit's presence in a trial with a baseline of the first {NUM_BASELINE_POINTS} timepoints subtracted"
        nwb.add_unit_column(name="trial_response_firing_rates",
                            description=firing_rate_description + " raw, NOT zscored")
        nwb.add_unit_column(name="trial_response_zscored",
                            description=firing_rate_description + " zscored")
        nwb.add_unit_column(name="saccade_zeta_scores",
                            description="zeta score for this unit, if this unit is a saccade responding unit")
        nwb.add_unit_column(name="probe_zeta_scores",
                            description="zeta score for this unit, if this unit is a probe responding unit")
        nwb.add_unit_column(
            name="r_p_peri_trials",
            description="Waveforms of each unit in each trial for mixed (perisaccadic) trials. "
                        + "Probe waveform responses have been subtracted by the average saccadic response for that unit"
                        + " (average across all saccadic trials) NOT zscored")
        nwb.add_unit_column(
            name="trial_spike_flags",
            description=f"Bool arr if the unit spiked in the trial duration ({TOTAL_TRIAL_MS}ms)"
        )

        threshold_fields = {}
        # helper func to pull out individual values for each unit into a kwargs dict for nwb

        def get_threshold_fields(unit_idxx: int) -> dict[str, bool]:
            vv = {}
            for k, v in threshold_fields.items():
                vv[k] = v[unit_idxx]
            return vv

        for threshold_name, (bool_list, desc) in self.unit_pop.unit_filters.items():
            th_name = f"threshold_{threshold_name}"
            nwb.add_unit_column(name=th_name, description=desc)
            threshold_fields[th_name] = bool_list

        # Get subtracted average saccade shifted responses for perisaccadic responses
        # (when probe and saccade, mixed)
        r_p_peri_trialdata = self.unit_pop.calc_rp_peri_trials()  # (trials, units, t)

        for unit_idx, unit_num in enumerate(self.unit_pop.unique_unit_nums):
            unit_spike_idxs = np.where(self.unit_pop.spike_clusters == unit_num)[0]
            spike_times = self.unit_pop.spike_timestamps[unit_spike_idxs]
            nwb.add_unit(
                spike_times=spike_times,
                trial_response_firing_rates=self.unit_pop.unit_firingrates[:, unit_idx],
                trial_response_zscored=self.unit_pop.unit_zscores[:, unit_idx],
                trial_spike_flags=self.unit_pop.trial_spike_flags[:, unit_idx],
                r_p_peri_trials=r_p_peri_trialdata[:, unit_idx],
                probe_zeta_scores=self.probe_zeta[unit_idx],
                saccade_zeta_scores=self.saccade_zeta[unit_idx],
                **get_threshold_fields(unit_idx)
            )

        # Add probe and saccade event timings, trial types
        behavior_events = nwb.create_processing_module(name="behavior",
                                                       description="Contains saccade and probe event timings")
        # Add metrics
        for metric_name, metric_data in self.metrics.items():
            ts = TimeSeries(name=f"metric-{metric_name}", data=metric_data, unit="num", rate=1.0, description=f"Quality metric {metric_name}")
            behavior_events.add(ts)

        # Misc data
        behavior_events.add(TimeSeries(name="unit-labels", data=self.unit_pop.unique_unit_nums, unit="num", rate=1.0, description="Unit number from kilosort for each unit"))
        behavior_events.add(TimeSeries(name="probes", data=self.probe_timestamps, unit="s", rate=0.001, description="Timestamps of the probe"))
        behavior_events.add(TimeSeries(name="saccades", data=self.saccade_timestamps, unit="s", rate=0.001, description="Timestamps of the saccades"))
        behavior_events.add(TimeSeries(name="spike_clusters", data=self.unit_pop.spike_clusters, unit="num", rate=1.0, description="Spike cluster assignments for the spike timings"))
        behavior_events.add(TimeSeries(name="spike_timestamps", data=self.unit_pop.spike_timestamps, unit="num", rate=1.0, description="Timestamps of each spike corresponding to the spike clusters"))
        behavior_events.add(TimeSeries(name="trial_event_idxs", data=self.unit_pop.get_trial_event_time_idxs(), unit="idx", rate=1.0, description="Index of each trial into the spike_timestamps"))
        behavior_events.add(TimeSeries(name="trial_durations_idxs", data=self.unit_pop.trial_durations_idxs.astype(int), unit="idxs", rate=1.0, description="Indexes of the start, stop for each trial, in terms of index into spike_times and spike_clusters"))
        behavior_events.add(TimeSeries(name="trial_motion_directions", data=self.unit_pop.get_trial_motion_directions(), unit="motion", rate=1.0, description="Motion direction of the drifting grating"))
        behavior_events.add(TimeSeries(name="trial_block_idx", data=self.unit_pop.get_trial_block_idx(), unit="idxs", rate=1.0, description="Which block of drifting grating did the trial occur in"))

        trial_types = np.array(self.unit_pop.get_trial_labels())
        unique_trial_types = np.unique(trial_types)
        for trial_type in unique_trial_types:
            behavior_events.add(TimeSeries(name=f"unit-trial-{trial_type}",
                                           data=np.where(trial_types == trial_type)[0], rate=1.0, unit="idx",
                                           description=f"Indices into all trials that are {trial_type} trials. Use nwbfile.units['trial_firing_rates'][unit_number][<idx goes here>] to get the firing rate of a unit in a given trial using these indicies"))

        # Want to specify when the saccade happened for the mixed trials (probe is always centered at 10ms)

        relative_saccade_times_for_mixed_trials = []

        for trial in self.unit_pop.get_mixed_trials():
            if trial.trial_label == "mixed":
                sac_idx = trial.events["saccade_event"]
                probe_idx = trial.events["probe_event"]
                sac_timestamp = self.unit_pop.spike_timestamps[sac_idx]
                probe_window_event_timestamp = self.unit_pop.spike_timestamps[probe_idx]
                relative_time = sac_timestamp - probe_window_event_timestamp
                relative_saccade_times_for_mixed_trials.append(relative_time)

        behavior_events.add(
            TimeSeries(
                name=f"mixed-trial-saccade-relative-timestamps",
                data=relative_saccade_times_for_mixed_trials, rate=0.001, unit="s",
                description=f"Timestamps of saccades in the mixed trials relative to the probe time"))

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

        # _testing = {"lower": [], "upper": [], "event": []}  # TODO comment me out

        other_len = len(other_timestamps)
        other_one_tenth = int(1 / 10 * other_len)
        for idx, ts in enumerate(other_timestamps):
            if idx % other_one_tenth == 0:
                print(f" {round(100 * (idx / other_len), 2)}%", end="")
            if np.isnan(ts):
                continue
            pre_window_ts = ts - (PRE_TRIAL_MS / 1000)
            post_window_ts = ts + (POST_TRIAL_MS / 1000)

            # 0th idx in tuple of np.where, 0th idx is the edge
            # which is the idx into spike_timestamps whose value is greater than our pre-window, so the lowest value
            # in spike_timestamps that is within our beginning window,
            start_idx = np.where(spike_timestamps >= pre_window_ts)[0][0]
            end_idx = np.where(spike_timestamps >= post_window_ts)[0][0]  # Same thing, lowest index where spike_timestamps is gte our post window
            ts_idx = np.where(spike_timestamps >= ts)[0][0]  # index of the lowest spike_timestamp that is gte our ts
            idx_ranges.append([start_idx, ts_idx, end_idx])
            tw = 2

        print("")
        # import matplotlib.pyplot as plt
        # plt.stairs(_testing["lower"])
        # plt.show()
        # plt.stairs(_testing["upper"])
        # plt.show()
        # plt.stairs(_testing["event"])
        # plt.show()
        tw = 2
        return idx_ranges

    def _demix_trials(self, saccade_idxs: list[list[float]], probe_idxs: list[list[float]]):
        # If a saccade and probe trial occur within +- .5 sec (500ms) then they should be considered a mixed trial
        # Indexes come in like [[start, event, stop], ..]

        saccade_ts = np.array([self.spike_timestamps[sp[1]] for sp in saccade_idxs])

        paired_passing = []  # [[probe_idx, [list of saccades, ..], ..]
        probe_failing = []  # [probe_idx, probe_idx, ..]

        for idx, data in enumerate(probe_idxs):
            _, event_idx, _ = data  # start, event, stop
            ts = self.spike_timestamps[event_idx]
            diff = saccade_ts - ts
            pos_res = diff <= MIXED_THRESHOLD  # Positive results
            if not np.any(pos_res):
                probe_failing.append(idx)
                continue
            pos_diff = diff[np.where(pos_res)]
            neg_res = pos_diff >= (-1 * MIXED_THRESHOLD)
            if np.any(neg_res):
                neg_res_idxs = np.where(neg_res)[0]  # Where the negative results are relative to the pos arr
                diff_idxs = np.where(pos_res)[0][neg_res_idxs]  # Where the results are (absolute diff idx)
                paired_passing.append([idx, diff_idxs])
            else:
                probe_failing.append(idx)
        probe_passing = np.array([r[0] for r in paired_passing])  # Array of indexes of passing probes [int, int, ..]
        saccade_passing = np.array([r[1][0] for r in paired_passing])  # array of passing saccades, only keeping first

        # [True/False, ...] where there were multiple saccades that were close to the probe
        multiple = np.array([len(r[1]) for r in paired_passing]) == 2
        # Invert to filter out multiple, going to just throw out those trials
        probe_passing_filtered = probe_passing[np.invert(multiple)]
        saccade_passing_filtered = saccade_passing[np.invert(multiple)]
        saccade_multi_filtered = saccade_passing[multiple]

        # Find the saccade_indexes that are NOT in the passing list, out of all possible indexes, by using set.difference
        sac_idx_set = set(range(len(saccade_idxs)))
        sac_filtered_idx_set = set(saccade_passing_filtered)
        # Remove passing (mixed) and multi passing (multiple saccades close to a probe)
        saccade_failing = list(sac_idx_set.difference(sac_filtered_idx_set).difference(saccade_multi_filtered))

        # probe_passing and saccade_passing are the same length, and are the mixed ones
        # saccade_failing is the 'pure' saccades
        # probe_failing is the 'pure' probes
        saccade_idxs = np.array(saccade_idxs)
        probe_idxs = np.array(probe_idxs)

        trials = {
            "saccade": saccade_idxs[saccade_failing],
            "probe": probe_idxs[probe_failing],
            "mixed": [{"probe": probe_idxs[probe_passing_filtered[i]], "saccade": saccade_idxs[saccade_passing_filtered[i]]} for i in range(len(probe_passing_filtered))]
        }

        # Convert the trial entries back into lists instead of numpy arrays
        for typ in ["saccade", "probe"]:
            val = trials[typ]
            newdata = []
            for trial in val:
                newdata.append(list(trial))
            trials[typ] = newdata

        # Convert the mixed as well
        newmixed = []
        for m in trials["mixed"]:
            newmixed.append({
                "probe": list(m["probe"]),
                "saccade": list(m["saccade"])
            })

        trials["mixed"] = newmixed
        return trials
