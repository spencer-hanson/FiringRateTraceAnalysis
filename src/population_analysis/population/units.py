# import numpy as np
# import numpy.ma as ma
# from population_analysis.consts import TOTAL_TRIAL_MS, SPIKE_BIN_MS, NUM_FIRINGRATE_SAMPLES, NUM_BASELINE_POINTS
# from population_analysis.processors.nwb.unit_preferred_direction import UnitPreferredDirection
#
#
# class Trial(object):
#     def __init__(self, start, end, trial_label, motion_direction, block_num):
#         self.start = start
#         self.end = end
#         self.trial_label = trial_label
#         self.motion_direction = motion_direction  # -1 or 1 (I don't remember which is which rn)
#         self.block_num = block_num  # Which block this trial resides in (should be 0-60)
#         self.events = {}
#
#     def __str__(self):
#         return f"Trial({self.trial_label}, start={self.start}, end={self.end}, events_count={len(list(self.events.items()))}, dir={self.motion_direction})"
#
#     def copy(self):
#         tr = Trial(self.start, self.end, self.trial_label, self.motion_direction, self.block_num)
#         tr.events = self.events.copy()
#         return tr
#
#     def add_event(self, time_idx, label):
#         self.events[label] = time_idx
#
#     @staticmethod
#     def create_saccade_trial(start, event, end, motion_direction, block_num):
#         t = Trial(start, end, "saccade", motion_direction, block_num)
#         t.add_event(event, "saccade_event")
#         return t
#
#     @staticmethod
#     def create_probe_trial(start, event, end, motion_direction, block_num):
#         t = Trial(start, end, "probe", motion_direction, block_num)
#         t.add_event(event, "probe_event")
#         return t
#
#     @staticmethod
#     def create_mixed_trial(saccade_start, saccade_event, saccade_end, probe_start, probe_event, probe_end, motion_direction, block_num):
#         # Disabled since trial is centered on probe
#         # start = saccade_start if saccade_start <= probe_start else probe_start
#         # end = saccade_end if saccade_end >= probe_end else probe_end
#
#         t = Trial(probe_start, probe_end, "mixed", motion_direction, block_num)  # Note the trial is centered on the probe regardless of the saccade timing
#         t.add_event(saccade_start, "saccade_start")
#         t.add_event(saccade_event, "saccade_event")
#         t.add_event(saccade_end, "saccade_end")
#
#         t.add_event(probe_start, "probe_start")
#         t.add_event(probe_event, "probe_event")
#         t.add_event(probe_end, "probe_end")
#
#         return t
#
#
# class UnitPopulation(object):
#     def __init__(self, spike_timestamps: np.ndarray, spike_clusters: np.ndarray, p_value_truth: np.ndarray):
#         self.spike_timestamps = spike_timestamps
#         self.spike_clusters = spike_clusters
#         self._trials = []
#         self._firing_rates = None  # Trial x UnitNum x time arr
#         self._trial_spike_flags = None  # Trial x UnitNum x time arr bool if it spiked or not
#         self._trial_durations_idxs = None  # Trial x 2 (start and stop indexes into spike_timestamps and clusters)
#         self._zscores = None
#         self._unit_filters = None
#         self._preferred_motions = None
#         self.unique_unit_nums = np.unique(spike_clusters)
#         self._num_units = len(self.unique_unit_nums)
#         self._p_value_truth = p_value_truth
#
#     def __str__(self):
#         return f"UnitPopulation(num_units={self._num_units}, num_trials={len(self._trials)})"
#
#     def get_mixed_trials(self):
#         trials = []
#         for tr in self._trials:
#             if tr.trial_label == "mixed":
#                 trials.append(tr.copy())
#         return trials
#
#     def add_saccade_trials(self, saccade_idx_list: list[list[int]]):
#         # Format [[start, event, end], ..] Where start is the index -200ms before the trial, event is the timestamps at
#         # and end is the idx 500ms after the trial
#         self._trials.extend([Trial.create_saccade_trial(*s) for s in saccade_idx_list])
#
#     def add_probe_trials(self, probe_idx_list: list[list[int]]):
#         # Format [[start, event, end], ..]
#         self._trials.extend([Trial.create_probe_trial(*p) for p in probe_idx_list])
#
#     def add_mixed_trials(self, mixed_list: list[dict[str, list[int]]]):
#         # Mixed in format [{"saccade": [start, event, end], "probe": ..}, ..]
#         self._trials.extend([
#             Trial.create_mixed_trial(*m["saccade"], *m["probe"])
#             for m in mixed_list
#         ])
#
#     def _zscore_unit_trial_waveforms(self, unit_trial_waveforms):
#         # unit_trial_waveforms is a (trials, units, 35) array
#         mean_baselines = np.mean(unit_trial_waveforms[:, :, :NUM_BASELINE_POINTS], axis=2)[:, :, None]  # Add axis for broadcasting
#         std_baselines = np.std(unit_trial_waveforms[:, :, :NUM_BASELINE_POINTS], axis=2)[:, :, None]  # Shape is trials x units x 1(new axis)
#
#         # Replace 0's with 1's so that the divide doesn't cause nan and inf TODO Fix?
#         np.place(std_baselines, std_baselines == 0, 1)
#
#         zscored = (unit_trial_waveforms - mean_baselines) / std_baselines
#         return zscored
#
#     def calc_rp_peri_trials(self):
#         # Want to calculate R_p(Peri) = R_mixed - R*_s
#         # Where R*_s is the time shifted average saccade response across the same units
#
#         # Grab the indexes of the saccades
#         saccade_trial_idxs = np.where(np.array(self.get_trial_labels()) == "saccade")[0]
#         saccade_unit_trial_waveforms = self.unit_firingrates[saccade_trial_idxs]  # trials x units x t(35)
#
#         # average is now units x t
#         saccade_unit_average_waveforms = np.average(saccade_unit_trial_waveforms, axis=0)  # Average over saccade trials for each unit
#         # Get mixed waveform unit-trials
#         mixed_trial_idxs = np.where(np.array(self.get_trial_labels()) == "mixed")[0]
#         mixed_unit_trial_waveforms = self.unit_firingrates[mixed_trial_idxs]  # trials x unit x t
#
#         # mixed_peri_waveforms is trials x units x t
#         # Reshape saccade_unit_average_waveforms to (1, units, t) and broadcast against (trials, units, t)
#         # to subtract saccade avg from all units in all trials
#         mixed_peri_waveforms = mixed_unit_trial_waveforms - saccade_unit_average_waveforms.reshape(1, *saccade_unit_average_waveforms.shape)
#
#         # Clamp negative values, cannot have a negative firing rate
#         mixed_peri_waveforms = np.clip(mixed_peri_waveforms, 0, None)
#
#         return mixed_peri_waveforms
#
#     def _gen_threshold_trials(self, fr):
#         # Remove units that do not meet a threshold firing rate across trials.
#         # Add filtering funcs here that should be pre-calculated
#
#         self._unit_filters = {
#             # Name of threshold: list of bool for each unit passing or not, str description
#             "zeta_passes": (self._p_value_truth, "True or false if the unit passes the zeta test for p < 0.01 for either saccade or probe")
#         }
#         return
#
#     def calc_firingrates(self):
#         # Calculate the firing rate of each unit for all trials
#         num_trials = len(self._trials)
#         firing_rates = np.empty((num_trials, self._num_units, NUM_FIRINGRATE_SAMPLES))
#         trial_spike_flags = np.full((num_trials, self._num_units, TOTAL_TRIAL_MS), fill_value=False, dtype="bool")
#         trial_durations_idxs = np.empty((num_trials, 2), dtype="int")
#
#         print("Calculating firing rates for all trials and units", end="")
#         one_tenth_of_trials = int(num_trials / 10)
#         for trial_idx, trial in enumerate(self._trials):
#
#             if trial_idx % one_tenth_of_trials == 0:
#                 print(f" {round(100 * (trial_idx/num_trials), 2)}%", end="")
#
#             trial_start = self.spike_timestamps[trial.start]
#             trial_end = self.spike_timestamps[trial.end]
#             # Make sure that the length of the trial is at least 700ms
#             trial_end = max(trial_end, trial_start + TOTAL_TRIAL_MS/1000)
#
#             trial_spike_clusters = self.spike_clusters[trial.start:trial.end]
#             trial_spike_times = self.spike_timestamps[trial.start:trial.end]
#             trial_durations_idxs[trial_idx, :] = np.array([trial.start, trial.end])
#
#             bins = np.arange(trial_start, trial_end + SPIKE_BIN_MS / 1000, SPIKE_BIN_MS / 1000)
#             bins = bins[:NUM_FIRINGRATE_SAMPLES + 1]  # Ensure that there are only 35 bins
#             spike_bins = np.arange(trial_start, trial_start + TOTAL_TRIAL_MS/1000 + .001, .001)[:TOTAL_TRIAL_MS + 1]
#
#             for unique_unit_num in range(self._num_units):
#                 single_unit_spike_times = trial_spike_times[np.where(trial_spike_clusters == unique_unit_num)]
#                 single_unit_firing_rate = np.histogram(
#                     single_unit_spike_times, bins=bins, density=False
#                 )[0] / SPIKE_BIN_MS  # Normalize by bin size
#
#                 firing_rates[trial_idx, unique_unit_num, :] = single_unit_firing_rate[:]
#
#                 # Get an array of size (700,) with counts for spikes at that ms, essentially single spike times
#                 # Histogram is a count of if the unit spiked in that time bin or not eg [0,0,0,0,1,0,1,0,0,..]
#                 single_unit_spike_array = np.histogram(single_unit_spike_times, bins=spike_bins, density=False)[0]
#                 single_unit_spike_array = np.logical_not(single_unit_spike_array == 0)
#                 trial_spike_flags[trial_idx, unique_unit_num, :] = single_unit_spike_array[:]
#                 tw = 2
#
#         # Mark units to be filtered out units using a threshold
#         self._gen_threshold_trials(firing_rates)
#         self._trial_durations_idxs = trial_durations_idxs
#         self._firing_rates = firing_rates  # (trials, units, t)
#         self._trial_spike_flags = trial_spike_flags
#         self._zscores = self._zscore_unit_trial_waveforms(firing_rates)
#         self._preferred_motions = UnitPreferredDirection(self._firing_rates.swapaxes(0, 1), self.get_trial_motion_directions()).calculate()
#
#         tw = 2
#         print("")
#
#     @property
#     def unit_firingrates(self):
#         if self._firing_rates is None:
#             self.calc_firingrates()
#         return self._firing_rates
#
#     @property
#     def trial_spike_flags(self):
#         if self._trial_spike_flags is None:
#             self.calc_firingrates()
#         return self._trial_spike_flags
#
#     @property
#     def trial_durations_idxs(self):
#         if self._trial_durations_idxs is None:
#             self.calc_firingrates()
#         return self._trial_durations_idxs
#
#     @property
#     def preferred_motions(self):
#         if self._preferred_motions is None:
#             self.calc_firingrates()
#         return self._preferred_motions
#
#     @property
#     def unit_zscores(self):
#         if self._zscores is None:
#             self.calc_firingrates()
#         return self._zscores
#
#     @property
#     def unit_filters(self):
#         if self._unit_filters is None:
#             self.calc_firingrates()
#         return self._unit_filters  # trials x units x t
#
#     def get_trial_labels(self):
#         return [tr.trial_label for tr in self._trials]
#
#     def get_trial_motion_directions(self):
#         return [tr.motion_direction for tr in self._trials]
#
#     def get_trial_block_idx(self):
#         return [tr.block_num for tr in self._trials]
#
#     def get_trial_event_time_idxs(self):
#         event_idxs = []
#         for trial in self._trials:
#             if trial.trial_label == "saccade":
#                 data = trial.events["saccade_event"]
#             else:
#                 data = trial.events["probe_event"]
#             event_idxs.append(data)
#         return np.array(event_idxs)
#
#     def get_trial_duration_event_idxs(self, offset=0):
#         # return a list like [[start, event, stop], ..] for all trials
#         data = []
#         for tr in self._trials:
#             if tr.trial_label == "saccade":
#                 event_time = tr.events["saccade_event"]
#             else:
#                 event_time = tr.events["probe_event"]
#
#             data.append([tr.start + offset, event_time + offset, tr.end + offset])
#         return data
