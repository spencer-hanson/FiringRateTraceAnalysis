import os

import numpy as np
from pynwb import NWBHDF5IO

from population_analysis.consts import TOTAL_TRIAL_MS, METRIC_NAMES, METRIC_THRESHOLDS
from population_analysis.processors.nwb.unit_filter import UnitFilter


class NWBSessionProcessor(object):
    def __init__(self, filepath_prefix_no_ext, filename, graph_folderpath, filter_mixed=True):
        filepath = f"{filepath_prefix_no_ext}/{filename}.nwb"
        graph_prefix = f"{graph_folderpath}/{filename}"

        if not os.path.exists(graph_prefix):
            os.makedirs(graph_prefix)

        nwbio = NWBHDF5IO(filepath)
        nwb = nwbio.read()

        self.probe_trial_idxs = nwb.processing["behavior"]["unit-trial-probe"].data[:]
        self.saccade_trial_idxs = nwb.processing["behavior"]["unit-trial-saccade"].data[:]
        self.mixed_trial_idxs = nwb.processing["behavior"]["unit-trial-mixed"].data[:]

        self.quality_metrics = self._extract_quality_metrics(nwb)

        # Filter out mixed trials that saccades are more than 20ms away from the probe
        self.mixed_rel_timestamps = nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]
        if filter_mixed:
            mixed_filtered_idxs = np.abs(self.mixed_rel_timestamps) <= 0.02  # only want mixed trials 20 ms within probe
            self.mixed_trial_idxs = self.mixed_trial_idxs[mixed_filtered_idxs]

        self.nwb = nwb
        tw = 2

    def _extract_quality_metrics(self, nwb):
        metrics = {}
        for metric_name in METRIC_NAMES.values():
            metrics[metric_name] = nwb.processing["behavior"][f"metric-{metric_name}"].data[:]
        return metrics

    @property
    def num_units(self):
        num_units = self.nwb.units["trial_spike_flags"].shape[0]  # units x trials x 700
        return num_units

    def probe_units(self):
        return self.nwb.units["trial_response_firing_rates"].data[:, self.probe_trial_idxs]

    def saccade_units(self):
        return self.nwb.units["trial_response_firing_rates"].data[:, self.saccade_trial_idxs]

    def mixed_units(self):
        return self.nwb.units["trial_response_firing_rates"].data[:, self.mixed_trial_idxs]

    def rp_peri_units(self):
        return self.nwb.units["r_p_peri_trials"].data[:]  # units x trials x t

    def units(self):
        return self.nwb.units["trial_response_firing_rates"].data[:]  # units x trials x t

    def qm_unit_filter(self) -> UnitFilter:

        def passing_quality_metrics(unit_num) -> bool:
            """
            My approach for handling the manually curated spikes is to pass everything through a 2-part filter.
            For the first part, I ask: What is the quality label? If it's 1 I let the unit through, if it's 0,
            I move on to step 2. For the second part, I ask: Does the unit meet spike-sorting quality metric thresholds.
            If yes, I let it through. If no, I exclude it.
            """
            label = self.quality_metrics["quality_labeling"][unit_num]
            if label == 1:
                return True

            for metric_name, metric_func in METRIC_THRESHOLDS.items():
                val = self.quality_metrics[metric_name][unit_num]
                result = metric_func(val)
                if not result:
                    return False

            return True

        return UnitFilter(passing_quality_metrics, self.num_units)

    def probe_zeta_unit_filter(self) -> UnitFilter:
        passing_idxs = np.where(self.nwb.units["probe_zeta_scores"][:] < 0.01)[0]
        passing_units = np.array(list(range(self.num_units)))[passing_idxs]

        def does_pass(unit_num):
            p = unit_num in passing_units
            return p

        return UnitFilter(does_pass, self.num_units)

    def activity_threshold_unit_filter(self, spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_zscore) -> UnitFilter:
        # Want units that spike at least <spike_count_threshold> times,
        # in at least trial_threshold % of the trials (NOTE CURRENTLY ONLY CHECKING Rp_Extra trials)
        # at most there can be missing_threshold% close to zero trials,
        # "close to zero trials" is any trial with less than <min_missing> spikes
        # baseline_zscore is the zscore of the average value of the first 8 timepoints is from
        # the trial average baseline first 8 timepoints
        baseline_mean = np.mean(np.mean(np.mean(self.units()[:, self.probe_trial_idxs, :][:, :, :8], axis=1), axis=0))
        baseline_std = np.std(np.mean(np.mean(self.units()[:, self.probe_trial_idxs, :][:, :, :8], axis=1), axis=1))

        def passing_activity(unit_num):
            bool_counts = self.nwb.units["trial_spike_flags"]  # units x trials x 700

            unit_trials = bool_counts[unit_num, :, :][self.probe_trial_idxs, :]  # trials x 700
            trial_count = len(self.probe_trial_idxs)

            trial_spike_sum = np.sum(unit_trials, axis=1)
            passing_trial_count = len(np.where(trial_spike_sum >= spike_count_threshold)[0])
            missing_trial_count = len(np.where(trial_spike_sum < min_missing)[0])

            condition = (trial_count * trial_threshold) <= passing_trial_count
            condition = ((trial_count * missing_threshold) >= missing_trial_count) and condition

            cur_mean = np.mean(np.mean(self.units()[unit_num][self.probe_trial_idxs, :][:, :8], axis=1), axis=0)
            zscore = abs((cur_mean - baseline_mean) / baseline_std)
            passes_zscore = zscore < baseline_zscore
            condition = condition and passes_zscore

            return condition

        u = UnitFilter(passing_activity, self.num_units)
        return u

    #
    # def activity_filtered_units_idxs(self, unit_filter=None):
    #     # unit_filter is an array of indexes to pre-filter the trial_spike_flags, return will be indexed relative
    #     bool_counts = self.nwb.units["trial_spike_flags"]  # units x trials x 700
    #     num_units = bool_counts.shape[0]
    #

    #     passing_units_idxs = []
    #
    #     for unit_num in range(num_units):
    #         condition = self.passing_activity(unit_num)
    #
    #         if condition:
    #             passing_units_idxs.append(unit_num)
    #
    #     passing = passing_units_idxs
    #     if unit_filter is not None:
    #         passing = sorted(list(set(unit_filter).intersection(passing)))
    #
    #     return passing
