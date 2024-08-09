import os.path

import h5py
import numpy as np
import pendulum
from pynwb import TimeSeries
from pynwb.file import Subject
from simply_nwb import SimpleNWB

from population_analysis.consts import MOUSE_DETAILS, METRIC_NAMES, UNIT_ZETA_P_VALUE, SESSION_DESCRIPTION, \
    EXPERIMENTERS, EXPERIMENT_DESCRIPTION, EXPERIMENT_KEYWORDS, DEVICE_NAME, DEVICE_DESCRIPTION, DEVICE_MANUFACTURER, \
    TOTAL_TRIAL_MS, NUM_BASELINE_POINTS, SPIKE_BIN_MS
from population_analysis.processors.experiments.saccadic_modulation import SaccadicModulationTrialProcessor
from population_analysis.processors.experiments.saccadic_modulation.firing_rates import FiringRateCalculator
from population_analysis.processors.experiments.saccadic_modulation.spikes import SpikeTrialOrganizer
from population_analysis.processors.kilosort import KilosortProcessor


class HDFSessionProcessor(object):
    def __init__(self, filename, mouse_name, session_id):
        self.raw_data = h5py.File(filename)
        self.session_id = session_id

        self.mouse_name = mouse_name
        self.mouse_name = mouse_name
        self.mouse_birthday = MOUSE_DETAILS[mouse_name]["birthday"]
        self.mouse_strain = MOUSE_DETAILS[mouse_name]["strain"]
        self.mouse_sex = MOUSE_DETAILS[mouse_name]["sex"]

        self.spike_clusters = np.array(self.raw_data["spikes"]["clusters"])
        self.spike_timings = np.array(self.raw_data["spikes"]["timestamps"])
        self.unique_unit_nums = np.unique(self.spike_clusters)

        self.probe_zeta = np.array(self.raw_data["zeta"]["probe"]["left"]["p"])
        self.saccade_zeta = np.array(self.raw_data["zeta"]["saccade"]["nasal"]["p"])

        self.metrics = self._extract_metrics(self.raw_data)
        self.p_value_truth = self._calc_p_value_truth(self.probe_zeta, self.saccade_zeta)

    def save_to_nwb(self, nwb_filename, load_precalculated=True):
        kp = KilosortProcessor(self.spike_clusters, self.spike_timings)

        raw_firing_rates, fr_bins = kp.calculate_firingrates(SPIKE_BIN_MS, load_precalculated)
        raw_spike_times = kp.calculate_spikes(load_precalculated)
        grating_windows = self._calc_grating_windows(self.raw_data)
        # Grab the trials for the events
        events = self._event_timings(self.raw_data, grating_windows)

        # Separate the trials into Rs, RpExtra and Rmixed
        smp = SaccadicModulationTrialProcessor(fr_bins, events)
        trialgroup = smp.calculate()
        trial_spike_duration_idxs = self._calc_trial_spike_duration_idxs(trialgroup)
        # debug check offset aa = [tr.events["saccade_time"] - tr.events["probe_time"] for tr in trialgroup.get_trials_by_type("mixed")]
        tfrc = FiringRateCalculator(raw_firing_rates, trialgroup)
        all_firing_rates = tfrc.calculate(load_precalculated)

        spike_organizer = SpikeTrialOrganizer(raw_spike_times, trialgroup)
        trial_spike_times = spike_organizer.calculate(load_precalculated)

        print("Creating NWB..")
        nwb = self._initialize_nwb()
        behavior_events = nwb.create_processing_module(name="behavior", description="Contains saccade and probe event timings")

        print("Adding firing rates and spikes..")
        self._add_rates_nwb(nwb, all_firing_rates, trial_spike_times, trial_spike_duration_idxs)

        print("Adding experiment and behavior data..")
        self._add_metrics_nwb(behavior_events)
        self._add_misc_data(behavior_events, events, trialgroup)
        self._add_trial_type_idxs(behavior_events, trialgroup)

        print("Writing to file (this may take a while)..")
        SimpleNWB.write(nwb, nwb_filename)
        print("Clearing memmaps..")
        raw_firing_rates._mmap.close()
        raw_spike_times._mmap.close()
        trial_spike_times._mmap.close()
        for _, val in all_firing_rates.items():
            val._mmap.close()
        del raw_spike_times
        del raw_firing_rates
        del trial_spike_times
        del all_firing_rates
        print("Done!")

    def _calc_trial_spike_duration_idxs(self, trialgroup):
        dur = []
        for tr in trialgroup.all_trials():
            dur.append([tr.start_idx*SPIKE_BIN_MS, tr.end_idx*SPIKE_BIN_MS])  # TODO more precisely calculate start and stop times for the trials
        return np.array(dur)

    def _add_rates_nwb(self, nwb, all_firing_rates, trial_spike_times, trial_spike_duration_idxs):
        datas = [
            ("large_range_normalized_firing_rates", all_firing_rates["largerange_normalized_firing_rate"]),
            ("trial_response_firing_rates", all_firing_rates["firing_rate"]),
            ("normalized_trial_response_firing_rates", all_firing_rates["normalized_firing_rate"]),
            ("trial_rp_peri_response_firing_rates", all_firing_rates["rp_peri_firing_rate"]),
            ("normalized_trial_rp_peri_response_firing_rates", all_firing_rates["rp_peri_normalized_firing_rate"]),
            ("trial_spike_times", trial_spike_times),
            ("unit_labels", self.unique_unit_nums),
            ("probe_zeta_scores", self.probe_zeta),
            ("saccade_zeta_scores", self.saccade_zeta),
            ("trial_spike_duration_idxs", trial_spike_duration_idxs)
        ]

        for idx, d in enumerate(datas):
            name, content = d
            print(f"Processing {name}..")
            nwb.processing["behavior"].add(TimeSeries(name=name, data=content, rate=1.0, unit="spikes", description=name))

    def _add_trial_type_idxs(self, behavior_events, trialgroup):
        trial_types = trialgroup.get_trials_attribute("trial_label")
        unique_trial_types = np.unique(trial_types)
        for trial_type in unique_trial_types:
            behavior_events.add(TimeSeries(name=f"unit-trial-{trial_type}",
                                           data=np.where(trial_types == trial_type)[0], rate=1.0, unit="idx",
                                           description=f"Indices into all trials that are {trial_type} trials."))
        # Relative timestamps for mixed trials
        mixed_trial_relative_timings = []
        for tr in trialgroup.get_trials_by_type("mixed"):
            mixed_trial_relative_timings.append(tr.events["saccade_time"] - tr.events["probe_time"])

        mixed_trial_relative_timings = np.array(mixed_trial_relative_timings)
        behavior_events.add(TimeSeries(
                name=f"mixed-trial-saccade-relative-timestamps",
                data=mixed_trial_relative_timings, rate=0.001, unit="s",
                description=f"Timestamps of saccades in the mixed trials relative to the probe time"))

    def _add_misc_data(self, behavior_events, events, trialgroup):
        # Misc data
        behavior_events.add(TimeSeries(name="probes", data=events["probe_timestamps"], unit="s", rate=0.001, description="Timestamps of the probe"))
        behavior_events.add(TimeSeries(name="saccades", data=events["saccade_timestamps"], unit="s", rate=0.001, description="Timestamps of the saccades"))

        behavior_events.add(TimeSeries(name="spike_clusters", data=self.spike_clusters, unit="num", rate=1.0, description="Spike cluster assignments for the spike timings"))
        behavior_events.add(TimeSeries(name="spike_timestamps", data=self.spike_timings, unit="num", rate=1.0, description="Timestamps of each spike corresponding to the spike clusters"))

        behavior_events.add(TimeSeries(name="trial_motion_directions", data=trialgroup.get_trials_attribute("motion_direction"), unit="motion", rate=1.0, description="Motion direction of the drifting grating"))
        behavior_events.add(TimeSeries(name="trial_block_idx", data=trialgroup.get_trials_attribute("block_idx"), unit="idxs", rate=1.0, description="Which block of drifting grating did the trial occur in"))

    def _add_metrics_nwb(self, event_module):
        for metric_name, metric_data in self.metrics.items():
            ts = TimeSeries(name=f"metric-{metric_name}", data=metric_data, unit="num", rate=1.0, description=f"Quality metric {metric_name}")
            event_module.add(ts)

    def _add_units_nwb(self, nwb, all_spike_times, all_firing_rates, trial_spike_times):
        for unit_idx, unit_num in enumerate(self.unique_unit_nums):
            print(f"Adding unit {unit_idx}/{len(self.unique_unit_nums)}..")
            nwb.add_unit(
                spike_times=all_spike_times[unit_idx, :],
                trial_spike_times=trial_spike_times,
                probe_zeta_scores=self.probe_zeta[unit_idx],
                saccade_zeta_scores=self.saccade_zeta[unit_idx],
                cluster_num=unit_num
            )
            tw = 2

    def _initialize_nwb(self):
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
            session_id=self.session_id,
            institution="CU Anschutz",
            keywords=EXPERIMENT_KEYWORDS,
            # related_publications="DOI::LINK GOES HERE FOR RELATED PUBLICATIONS"
        )

        # Add device
        nwb.create_device(
            name=DEVICE_NAME, description=DEVICE_DESCRIPTION, manufacturer=DEVICE_MANUFACTURER
        )

        # Add units
        # nwb.add_unit_column(name="trial_spike_times", description=f"Arr of the units spikes in the trial duration ({TOTAL_TRIAL_MS}ms)")
        #
        # nwb.add_unit_column(name="saccade_zeta_scores", description="zeta score for this unit, if this unit is a saccade responding unit")
        # nwb.add_unit_column(name="probe_zeta_scores", description="zeta score for this unit, if this unit is a probe responding unit")
        #
        # nwb.add_unit_column(name="cluster_num", description="Cluster ID for the unit")


        #
        # firing_rate_description = f"trials x response time length array for each unit"
        # nwb.add_unit_column(name="trial_response_firing_rates", description=firing_rate_description)
        # nwb.add_unit_column(name="normalized_trial_response_firing_rates", description=firing_rate_description + ", normalized by subtracting a baseline and dividing by the standard deviation of the baseline")
        # nwb.add_unit_column(name="trial_rp_peri_response_firing_rates",
        #                     description="Normalized responses of each unit in each trial for mixed (perisaccadic) trials. "
        #                 + "Probe waveform responses have been subtracted by the average saccadic response for that unit"
        #                 + " (average across all saccadic trials)")
        #
        # nwb.add_unit_column(name="normalized_trial_rp_peri_response_firing_rates",
        #                     description="Normalized responses of each unit in each trial for mixed (perisaccadic) trials. "
        #                                 + "Probe waveform responses have been subtracted by the average saccadic response for that unit"
        #                                 + " (average across all saccadic trials)")
        #

        return nwb

    def _calc_p_value_truth(self, probe_zeta, saccade_zeta):
        # Calculate a bool array of if the units pass the p-value zeta test
        probe = probe_zeta <= UNIT_ZETA_P_VALUE
        saccade = saccade_zeta <= UNIT_ZETA_P_VALUE
        combined = np.logical_or(probe, saccade)  # A unit can pass probe or saccade to be included
        return combined

    def _extract_metrics(self, hd5data):
        metrics = {}
        for k, v in METRIC_NAMES.items():
            metrics[v] = np.array(hd5data["metrics"][k])
        return metrics

    def _calc_grating_windows(self, raw_data):
        # zip up timestamps of the drifting grating with the corresponding iti (inter time intervals) to give [[grating_start, grating_stop], ..]
        window_timestamps = np.array(list(zip(list(raw_data["stimuli"]["dg"]["grating"]["timestamps"]),
                                              list(raw_data["stimuli"]["dg"]["iti"]["timestamps"]))))
        inter_grating_timestamps = []
        for i in range(1, len(window_timestamps)):
            _, last_stop = window_timestamps[i - 1]
            next_start, _ = window_timestamps[i]
            inter_grating_timestamps.append([last_stop, next_start])

        inter_grating_timestamps = np.array(
            inter_grating_timestamps)  # Timestamps of no motion of the drifting grating in [[dg_stop, dg_start], ..]

        return {
            "grating_timestamps": window_timestamps,
            "inter_grating_timestamps": inter_grating_timestamps
        }

    def _event_timings(self, raw_data, grating_windows):
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

        probe_timestamps, probe_motions, probe_blocks = self._calc_motions(probe_timestamps, grating_windows,
                                                                      grating_motion_directions, "probes")
        saccade_timestamps, saccade_motions, saccade_blocks = self._calc_motions(saccade_timestamps, grating_windows,
                                                                            grating_motion_directions, "saccades")

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

    def _calc_motions(self, timestamps, windows, motions, name):
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

