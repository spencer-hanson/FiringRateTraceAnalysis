import numpy as np

from population_analysis.consts import TOTAL_TRIAL_MS, SPIKE_BIN_MS


class Trial(object):
    def __init__(self, start, end, trial_label):
        self.start = start
        self.end = end
        self.trial_label = trial_label
        self.events = {}

    def __str__(self):
        return f"Trial({self.start}, {self.end}, events_count={len(list(self.events.items()))})"

    def add_event(self, timestamp, label):
        self.events[label] = timestamp

    @staticmethod
    def create_saccade_trial(start, event, end):
        t = Trial(start, end, "saccade")
        t.add_event(event, "saccade")
        return t

    @staticmethod
    def create_probe_trial(start, event, end):
        t = Trial(start, end, "probe")
        t.add_event(event, "probe")
        return t

    @staticmethod
    def create_mixed_trial(saccade_start, saccade_event, saccade_end, probe_start, probe_event, probe_end):
        # Disabled since trial is centered on probe
        # start = saccade_start if saccade_start <= probe_start else probe_start
        # end = saccade_end if saccade_end >= probe_end else probe_end

        t = Trial(probe_start, probe_end, "mixed")  # Note the trial is centered on the probe regardless of the saccade timing
        t.add_event(saccade_start, "saccade_start")
        t.add_event(saccade_event, "saccade_event")
        t.add_event(saccade_end, "saccade_end")

        t.add_event(probe_start, "probe_start")
        t.add_event(probe_event, "probe_event")
        t.add_event(probe_end, "probe_end")

        return t


class UnitPopulation(object):
    def __init__(self, spike_timestamps: np.ndarray, spike_clusters: np.ndarray):
        self.spike_timestamps = spike_timestamps
        self.spike_clusters = spike_clusters
        self._trials = []
        self._firing_rates = None  # UnitNum x Trial array
        self.num_units = len(np.unique(self.spike_clusters))

    def __str__(self):
        return f"UnitPopulation(num_units={self.num_units}, num_trials={len(self._trials)})"

    def add_saccade_trials(self, saccade_idx_list: list[list[int]]):
        # Format [[start, event, end], ..] Where start is the index -200ms before the trial, event is the timestamps at
        # and end is the idx 500ms after the trial
        self._trials.extend([Trial.create_saccade_trial(s[0], s[1], s[2]) for s in saccade_idx_list])

    def add_probe_trials(self, probe_idx_list: list[list[int]]):
        # Format [[start, event, end], ..]
        self._trials.extend([Trial.create_probe_trial(p[0], p[1], p[2]) for p in probe_idx_list])

    def add_mixed_trials(self, mixed_list: list[dict[str, list[int]]]):
        # Mixed in format [{"saccade": [start, event, end], "probe": ..}, ..]
        self._trials.extend([
            Trial.create_mixed_trial(*m["saccade"], *m["probe"])
            for m in mixed_list
        ])

    def calc_firingrates(self):
        # Calculate the firing rate of each unit for all trials
        # for trial in self._trials:
        #     trial_spike_times = self.spike_timestamps[trial.start:trial.end]
        #     trial_start = self.spike_timestamps[trial.start]
        #     spike_time_index = 0
        #     trial_spike_clusters = []
        #
        #     bins = np.arange(trial_start, self.spike_timestamps[trial.end], SPIKE_BIN_MS/1000)
        #     digs = np.digitize(trial_spike_times, bins)
        #     firingrate = np.unique(digs, return_counts=True)[1] / SPIKE_BIN_MS
        #
        #     unique_units = np.unique(trial_spike_clusters)
        #
        #     for unit in range(self.num_units):
        #         if unit in unique_units:
        #             unit_spikes = np.where(trial_spike_clusters == unit)
        #             []
        #             pass
        #         else:
        #             trial_unit_spikes[unit, :] = 0
        #     tw = 2

        self._firing_rates = 0  # UnitNum x Trials array
        pass

    @property
    def unit_firingrates(self):
        if self._firing_rates is None:
            self.calc_firingrates()
        return self._firing_rates
