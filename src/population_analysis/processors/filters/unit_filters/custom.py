import numpy as np

from population_analysis.processors.filters.unit_filters import UnitFilter


class CustomUnitFilter(UnitFilter):
    def __init__(self, spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore, trial_spike_flags, units, probe_trial_idxs, num_units):
        self.spike_count_threshold = spike_count_threshold
        self.trial_threshold = trial_threshold
        self.missing_threshold = missing_threshold
        self.min_missing = min_missing
        self.baseline_mean_zscore = baseline_mean_zscore
        self.baseline_time_std_zscore = baseline_time_std_zscore
        # passing_func(unit_num) -> bool

        self.trial_spike_flags = trial_spike_flags
        self.units = units
        self.probe_trial_idxs = probe_trial_idxs
        self.num_units = num_units

        passing_func = self.get_passing_func()

        super().__init__(passing_func, num_units)

    def get_passing_func(self):
        def activity_threshold_unit_filter(unit_num):
            """
                Folder formatting name params guide:

                spike_count_threshold=sp,
                trial_threshold=tr,
                missing_threshold=ms,
                min_missing=mn,
                baseline_mean_zscore=bzm,
                baseline_time_std_zscore=bzs

                Want units that spike at least <spike_count_threshold> times,
                in at least trial_threshold % of the trials (NOTE CURRENTLY ONLY CHECKING Rp_Extra trials)
                at most there can be missing_threshold% close to zero trials,
                "close to zero trials" is any trial with less than <min_missing> spikes
                param baseline_mean_zscore:
                The zscore of the mean of the first 8 timepoints across trials and of the same units needs to be strictly less than this
                param baseline_time_std_zscore
                The zscore of the std of the first 8 timepoints, across trials of the same units needs to be strictly less than this
            """

            # baseline_mean = np.mean(np.mean(np.mean(self.units()[:, self.probe_trial_idxs, :][:, :, :8], axis=1), axis=0))
            # baseline_std = np.std(np.mean(np.mean(self.units()[:, self.probe_trial_idxs, :][:, :, :8], axis=1), axis=1))

            # _baseline_time_stds = np.mean(np.std(self.units()[:, self.probe_trial_idxs][:, :, :8], axis=2), axis=1)
            # The trial-averaged mean of the baseline timepoints stds, then unit averaged
            # baseline_time_std_mean = np.mean(_baseline_time_stds)
            # standard deviation of the baseline's trial-averaged standard deviations across units
            # baseline_time_std_std = np.std(_baseline_time_stds)

            bool_counts = self.trial_spike_flags  # units x trials x 700

            unit_trials = bool_counts[unit_num, :, :][self.probe_trial_idxs, :]  # trials x 700
            trial_count = len(self.probe_trial_idxs)

            trial_spike_sum = np.sum(unit_trials, axis=1)
            passing_trial_count = len(np.where(trial_spike_sum >= self.spike_count_threshold)[0])
            missing_trial_count = len(np.where(trial_spike_sum < self.min_missing)[0])

            condition = (trial_count * self.trial_threshold) <= passing_trial_count
            condition = ((trial_count * self.missing_threshold) >= missing_trial_count) and condition

            # Check that the mean of the baseline is within baseline_mean_zscore std's
            cur_mean = np.mean(np.mean(self.units[unit_num][self.probe_trial_idxs, :][:, :8], axis=1), axis=0)
            baseline_mean = np.mean(np.mean(self.units[unit_num][:, :8], axis=1), axis=0)
            baseline_std = np.mean(np.std(self.units[unit_num][:, :8], axis=1))
            baseline_std = baseline_std if baseline_std != 0 else 1

            mean_zscore = abs((cur_mean - baseline_mean) / baseline_std)
            condition = condition and mean_zscore < self.baseline_mean_zscore

            # Check that the standard deviation of the current unit's baseline over time
            # is at most baseline_time_std_zscore stds from the average std
            cur_time_std_mean = np.mean(np.std(self.units[unit_num][self.probe_trial_idxs][:, :8], axis=1))
            baseline_time_std_std = np.std(np.std(self.units[unit_num][:, :8], axis=1))
            baseline_time_std_std = baseline_time_std_std if baseline_time_std_std != 0 else 1

            std_zscore = (baseline_std - cur_time_std_mean) / baseline_time_std_std
            std_zscore = abs(std_zscore)
            condition = condition and std_zscore < self.baseline_time_std_zscore

            return condition

        return activity_threshold_unit_filter

    def get_basename(self):
        return f"qm_zeta_activity_{self.spike_count_threshold}sp_{self.trial_threshold}tr_{self.missing_threshold}ms_{self.min_missing}mn_{self.baseline_mean_zscore}bzm_{self.baseline_time_std_zscore}bzs"

