import os

import numpy as np
from pynwb import NWBHDF5IO

from population_analysis.consts import METRIC_NAMES
from population_analysis.processors.nwb.filters.trial_filters import TrialFilter
from population_analysis.processors.nwb.filters.trial_filters.motiondir import MotionDirectionTrialFilter
from population_analysis.processors.nwb.filters.unit_filters import UnitFilter

from population_analysis.processors.nwb.filters.unit_filters import CustomUnitFilter
from population_analysis.processors.nwb.filters.unit_filters import QualityMetricsUnitFilter
from population_analysis.processors.nwb.filters.unit_filters import ZetaUnitFilter


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
            self.mixed_filtered_idxs = np.where(mixed_filtered_idxs)[0]
        else:
            self.mixed_filtered_idxs = np.array([True] * len(self.mixed_rel_timestamps))

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

    @property
    def num_trials(self):
        num_trials = self.nwb.units["trial_spike_flags"].shape[1]
        return num_trials

    def spikes(self):
        return self.nwb.units["trial_spike_flags"]  # (units, trials, 700)

    def trial_motion_directions(self):
        return self.nwb.processing["behavior"]["trial_motion_directions"].data[:]

    def trial_block_idxs(self):
        return self.nwb.processing["behavior"]["trial_block_idx"].data[:]

    def trial_durations(self):
        return self.nwb.processing["behavior"]["trial_durations_idxs"].data[:]  # (trials, 2)  [start, stop]

    def probe_units(self):
        return self.nwb.units["trial_response_firing_rates"].data[:, self.probe_trial_idxs]

    def saccade_units(self):
        return self.nwb.units["trial_response_firing_rates"].data[:, self.saccade_trial_idxs]

    def mixed_units(self):
        return self.nwb.units["trial_response_firing_rates"].data[:, self.mixed_trial_idxs]

    def rp_peri_units(self):
        return self.nwb.units["r_p_peri_trials"].data[:][:, self.mixed_filtered_idxs]  # units x trials x t

    def units(self):
        return self.nwb.processing["behavior"]["units_normalized"].data[:].swapaxes(0, 1)
        # return self.nwb.units["trial_response_firing_rates"].data[:]  # units x trials x t


    def unit_filter_qm(self) -> UnitFilter:
        return QualityMetricsUnitFilter(self.quality_metrics, self.num_units)
    
    def unit_filter_probe_zeta(self) -> UnitFilter:
        return ZetaUnitFilter(self.nwb.units["probe_zeta_scores"][:])

    def unit_filter_custom(self, spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore) -> UnitFilter:
        return CustomUnitFilter(spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore, self.nwb.units["trial_spike_flags"], self.units(), self.probe_trial_idxs, self.num_units)

    def trial_motion_filter(self, motion_direction) -> TrialFilter:
        return MotionDirectionTrialFilter(motion_direction, self.trial_motion_directions())

