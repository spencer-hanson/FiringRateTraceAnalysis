import numpy as np
import numpy.ma as ma
from population_analysis.consts import TOTAL_TRIAL_MS, SPIKE_BIN_MS, NUM_FIRINGRATE_SAMPLES, NUM_BASELINE_POINTS


class Trial(object):
    def __init__(self, start, end, trial_label):
        self.start = start
        self.end = end
        self.trial_label = trial_label
        self.events = {}

    def __str__(self):
        return f"Trial({self.trial_label}, start={self.start}, end={self.end}, events_count={len(list(self.events.items()))})"

    def copy(self):
        tr = Trial(self.start, self.end, self.trial_label)
        tr.events = self.events.copy()
        return tr

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
    def __init__(self, spike_timestamps: np.ndarray, spike_clusters: np.ndarray, p_value_truth: np.ndarray):
        self.spike_timestamps = spike_timestamps
        self.spike_clusters = spike_clusters
        self._trials = []
        self._firing_rates = None  # Trial x UnitNum x time arr
        self._trial_spike_flags = None  # Trial x UnitNum x time arr bool if it spiked or not
        self._trial_durations_idxs = None  # Trial x 2 (start and stop indexes into spike_timestamps and clusters)
        self._zscores = None
        self.unique_spike_clusters = np.unique(self.spike_clusters)
        self._num_prefiltered_units = len(self.unique_spike_clusters)
        self._unit_filters = None
        self.unique_unit_nums = np.unique(spike_clusters)
        self._p_value_truth = p_value_truth

    def __str__(self):
        return f"UnitPopulation(num_units={self._num_prefiltered_units}, num_trials={len(self._trials)})"

    def get_mixed_trials(self):
        trials = []
        for tr in self._trials:
            if tr.trial_label == "mixed":
                trials.append(tr.copy())
        return trials

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

    def _zscore_unit_trial_waveforms(self, unit_trial_waveforms):
        # unit_trial_waveforms is a (trials, units, 35) array
        mean_baselines = np.mean(unit_trial_waveforms[:, :, :NUM_BASELINE_POINTS], axis=2)[:, :, None]  # Add axis for broadcasting
        std_baselines = np.std(unit_trial_waveforms[:, :, :NUM_BASELINE_POINTS], axis=2)[:, :, None]  # Shape is trials x units x 1(new axis)

        # Replace 0's with 1's so that the divide doesn't cause nan and inf TODO Fix?
        np.place(std_baselines, std_baselines == 0, 1)

        zscored = (unit_trial_waveforms - mean_baselines) / std_baselines
        return zscored

    def calc_rp_peri_trials(self):
        # Want to calculate R_p(Peri) = R_mixed - R*_s
        # Where R*_s is the time shifted average saccade response across the same units

        # Grab the indexes of the saccades
        saccade_trial_idxs = np.where(np.array(self.get_trial_labels()) == "saccade")[0]
        saccade_unit_trial_waveforms = self.unit_firingrates[saccade_trial_idxs]  # trials x units x t(35)

        # average is now units x t
        saccade_unit_average_waveforms = np.average(saccade_unit_trial_waveforms, axis=0)  # Average over saccade trials for each unit
        # Get mixed waveform unit-trials
        mixed_trial_idxs = np.where(np.array(self.get_trial_labels()) == "mixed")[0]
        mixed_unit_trial_waveforms = self.unit_firingrates[mixed_trial_idxs]  # trials x unit x t

        # mixed_peri_waveforms is trials x units x t
        # Reshape saccade_unit_average_waveforms to (1, units, t) and broadcast against (trials, units, t)
        # to subtract saccade avg from all units in all trials
        mixed_peri_waveforms = mixed_unit_trial_waveforms - saccade_unit_average_waveforms.reshape(1, *saccade_unit_average_waveforms.shape)

        # Clamp negative values, cannot have a negative firing rate
        mixed_peri_waveforms = np.clip(mixed_peri_waveforms, 0, None)

        return mixed_peri_waveforms

    def _gen_threshold_trials(self, fr):
        # Remove units that do not meet a threshold firing rate across trials.

        self._unit_filters = {
            # Name of threshold: list of bool for each unit passing or not, str description
            "zeta_passes": (self._p_value_truth, "True or false if the unit passes the zeta test for p < 0.01 for either saccade or probe")
        }
        return

        # Old thresholding code
        # from population_analysis.consts import TRIAL_THRESHOLD_SUM, UNIT_TRIAL_PERCENTAGE
        #
        # # must have at least a total of 0.01 firing rate in 20% of the trials of Saccade OR Probe trials
        # # OR is used to include units that are only responsive in one trial
        # units = firing_rates.swapaxes(0, 1)  # Swap trials and units
        # # units arr should be units x trials x t
        # num_trials = units.shape[1]
        # # The unit must have at least 0.01 activity on average in 20% of the trials
        # # divide by 2 to account for units that only fire in saccade or probe trials
        # threshold = TRIAL_THRESHOLD_SUM * (num_trials/2) * UNIT_TRIAL_PERCENTAGE
        # unit_all_trial_activity = np.sum(np.sum(units, axis=2), axis=1)  # Sum across all trials of the same unit, all responses
        # unit_mask = unit_all_trial_activity > threshold
        # unit_idxs = np.where(unit_mask)
        # new_units = units[unit_idxs]
        # self._filtered_unit_nums = self.unique_unit_nums[unit_idxs]
        # new_firing_rates = new_units.swapaxes(0, 1)  # Swap units and trials back to that it is (trials, units, t)
        # return new_firing_rates

    def calc_firingrates(self):
        # Calculate the firing rate of each unit for all trials
        num_trials = len(self._trials)
        firing_rates = np.empty((num_trials, self._num_prefiltered_units, NUM_FIRINGRATE_SAMPLES))
        trial_spike_flags = np.empty((num_trials, self._num_prefiltered_units, TOTAL_TRIAL_MS), dtype="bool")
        trial_durations_idxs = np.empty((num_trials, 2), dtype="int")

        print("Calculating firing rates for all trials and units", end="")
        one_tenth_of_trials = int(num_trials / 10)
        for trial_idx, trial in enumerate(self._trials):
            trial_durations_idxs[trial_idx, :] = np.array([trial.start, trial.end])

            if trial_idx % one_tenth_of_trials == 0:
                print(f" {round(100 * (trial_idx/num_trials), 2)}%", end="")

            trial_spike_times = self.spike_timestamps[trial.start:trial.end]
            trial_start = self.spike_timestamps[trial.start]
            trial_end = self.spike_timestamps[trial.end]
            # Make sure that the length of the trial is at least 700ms
            trial_end = max(trial_end, trial_start + TOTAL_TRIAL_MS/1000)
            trial_spike_clusters = self.spike_clusters[trial.start:trial.end]

            # all_units_mask is (units, num_spikes)
            # Takes the spikes (num_spikes,) and broadcasts it to (units, num_spikes) so each unit has a copy of
            # which spikes belong to which unit
            all_units_mask = np.broadcast_to(trial_spike_clusters[:, None].T,
                                             (len(self.unique_unit_nums), len(trial_spike_clusters)))
            # Then mask out each
            all_units_mask = all_units_mask == self.unique_unit_nums[:, None]  # Mask on trial for each unique value

            unmasked_spike_times = np.broadcast_to(trial_spike_times, (len(self.unique_unit_nums), *trial_spike_times.shape))
            # Mask out the times where the spike time doesn't belong to each spike
            unique_unit_spike_times = ma.array(unmasked_spike_times, mask=~all_units_mask)

            bins = np.arange(trial_start, trial_end + SPIKE_BIN_MS / 1000, SPIKE_BIN_MS / 1000)
            bins = bins[:NUM_FIRINGRATE_SAMPLES + 1]  # Ensure that there are only 35 bins
            spike_bins = np.arange(trial_start, trial_start + TOTAL_TRIAL_MS/1000 + .001, .001)[:TOTAL_TRIAL_MS + 1]

            for unique_unit_num in range(self._num_prefiltered_units):
                single_unit_spike_times = unique_unit_spike_times[unique_unit_num]
                single_unit_spike_times = single_unit_spike_times.compressed()
                single_unit_firing_rate = np.histogram(
                    single_unit_spike_times, bins=bins, density=False
                )[0] / SPIKE_BIN_MS  # Normalize by bin size
                # Calculate the absolute unit index (not used anymore since all units are included)
                # absolute_unit_idx = np.where(self.unique_spike_clusters == self.unique_unit_nums[unique_unit_num])[0][0]
                firing_rates[trial_idx, unique_unit_num, :] = single_unit_firing_rate[:]
                
                # Get an array of size (700,) with counts for spikes at that ms, essentially single spike times
                single_unit_spike_array = np.histogram(single_unit_spike_times, bins=spike_bins, density=False)[0]
                single_unit_spike_array = np.logical_not(single_unit_spike_array == 0)
                trial_spike_flags[trial_idx, unique_unit_num, :] = single_unit_spike_array[:]

        # Mark units to be filtered out units using a threshold
        self._gen_threshold_trials(firing_rates)
        self._trial_durations_idxs = trial_durations_idxs
        self._firing_rates = firing_rates  # (trials, units, t)
        self._trial_spike_flags = trial_spike_flags
        self._zscores = self._zscore_unit_trial_waveforms(firing_rates)
        tw = 2
        print("")

    @property
    def unit_firingrates(self):
        if self._firing_rates is None:
            self.calc_firingrates()
        return self._firing_rates

    @property
    def trial_spike_flags(self):
        if self._trial_spike_flags is None:
            self.calc_firingrates()
        return self._trial_spike_flags

    @property
    def trial_durations_idxs(self):
        if self._trial_durations_idxs is None:
            self.calc_firingrates()
        return self._trial_durations_idxs

    @property
    def unit_zscores(self):
        if self._zscores is None:
            self.calc_firingrates()
        return self._zscores

    @property
    def unit_filters(self):
        if self._unit_filters is None:
            self.calc_firingrates()
        return self._unit_filters  # trials x units x t

    def get_trial_labels(self):
        return [tr.trial_label for tr in self._trials]

