import numpy as np
import numpy.ma as ma
from population_analysis.consts import TOTAL_TRIAL_MS, SPIKE_BIN_MS, NUM_FIRINGRATE_SAMPLES


class Trial(object):
    def __init__(self, start, end, trial_label):
        self.start = start
        self.end = end
        self.trial_label = trial_label
        self.events = {}

    def __str__(self):
        return f"Trial({self.trial_label}, start={self.start}, end={self.end}, events_count={len(list(self.events.items()))})"

    def add_event(self, timestamp, label):
        self.events[label] = timestamp

    @staticmethod
    def create_saccade_trial(start, event, end):
        t = Trial(start, end, "saccade")
        t.add_event(event, "saccade_event")
        return t

    @staticmethod
    def create_probe_trial(start, event, end):
        t = Trial(start, end, "probe")
        t.add_event(event, "probe_event")
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
        self._firing_rates = None  # UnitNum x Trial x time arr
        self.unique_spike_clusters = np.unique(self.spike_clusters)
        self.num_units = len(self.unique_spike_clusters)

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
        num_trials = len(self._trials)
        firing_rates = np.empty((num_trials, self.num_units, NUM_FIRINGRATE_SAMPLES))

        print("Calculating firing rates for all trials and units", end="")
        one_tenth_of_trials = int(num_trials / 10)
        for trial_idx, trial in enumerate(self._trials):
            if trial_idx % one_tenth_of_trials == 0:
                print(f" {round(trial_idx/num_trials, 2)} %", end="")

            trial_spike_times = self.spike_timestamps[trial.start:trial.end]
            trial_start = self.spike_timestamps[trial.start]
            # Make sure that the length of the trial is at least 700ms
            trial_end = max(self.spike_timestamps[trial.end], trial_start + TOTAL_TRIAL_MS/1000)
            trial_spike_clusters = self.spike_clusters[trial.start:trial.end]

            unique_units = np.unique(trial_spike_clusters)

            all_units_mask = np.broadcast_to(trial_spike_clusters[:, None].T,
                                             (len(unique_units), len(trial_spike_clusters)))
            all_units_mask = all_units_mask == unique_units[:, None]  # Mask on trial for each unique value

            unmasked_spike_times = np.broadcast_to(trial_spike_times, (len(unique_units), *trial_spike_times.shape))
            # Mask out the times where the spike time doesn't belong to each spike
            unique_unit_spike_times = ma.array(unmasked_spike_times, mask=~all_units_mask)

            bins = np.arange(trial_start, trial_end + SPIKE_BIN_MS / 1000, SPIKE_BIN_MS / 1000)
            bins = bins[:NUM_FIRINGRATE_SAMPLES + 1]  # Ensure that there are only 35 bins

            num_unique_units = len(unique_units)
            abs_unit_idxs = []
            for unique_unit_num in range(num_unique_units):
                single_unit_spike_times = unique_unit_spike_times[unique_unit_num]
                single_unit_spike_times = single_unit_spike_times.compressed()
                single_unit_firing_rate = np.histogram(
                    single_unit_spike_times, bins=bins, density=False
                )[0] / SPIKE_BIN_MS  # Normalize by bin size
                # Calculate the absolute unit index
                absolute_unit_idx = np.where(self.unique_spike_clusters == unique_units[unique_unit_num])[0][0]
                abs_unit_idxs.append(absolute_unit_idx)
                firing_rates[trial_idx, absolute_unit_idx, :] = single_unit_firing_rate[:]

        self._firing_rates = firing_rates
        print("")

    @property
    def unit_firingrates(self):
        if self._firing_rates is None:
            self.calc_firingrates()
        return self._firing_rates

    def get_trial_labels(self):
        return [tr.trial_label for tr in self._trials]

