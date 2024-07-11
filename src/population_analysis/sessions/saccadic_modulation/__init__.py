import numpy as np
from pynwb import NWBHDF5IO

from population_analysis.consts import METRIC_NAMES
from population_analysis.processors.filters import BasicFilter
from population_analysis.processors.filters.trial_filters import TrialFilter
from population_analysis.processors.filters.trial_filters.motiondir import MotionDirectionTrialFilter
from population_analysis.processors.filters.trial_filters.rp_peri import RelativeTrialFilter
from population_analysis.processors.filters.unit_filters import UnitFilter

from population_analysis.processors.filters.unit_filters import CustomUnitFilter
from population_analysis.processors.filters.unit_filters import QualityMetricsUnitFilter
from population_analysis.processors.filters.unit_filters import ZetaUnitFilter
from population_analysis.processors.experiments.saccadic_modulation.rp_peri_calculator import RpPeriCalculator


class NWBSession(object):
    def __init__(self, filepath_prefix_no_ext, filename, graph_folderpath=None, filter_mixed=True, use_normalized_units=True):
        filepath = f"{filepath_prefix_no_ext}/{filename}.nwb"
        self.filename_no_ext = filename
        self.filepath_prefix_no_ext = filepath_prefix_no_ext
        graph_prefix = f"{graph_folderpath}/{filename}"

        if filename.endswith(".nwb"):
            print("FILENAME SHOULD NOT END WITH .nwb in args (on disk todo fix this) remove in str")
        # if not os.path.exists(graph_prefix):  # TODO
        #     os.makedirs(graph_prefix)

        nwbio = NWBHDF5IO(filepath)
        nwb = nwbio.read()

        self.use_normalized_units = use_normalized_units
        self._normalized_rp_peri = None
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

        self.num_trials = self.nwb.processing["behavior"]["trial_motion_directions"].data[:].shape[0]
        self.num_units = self.nwb.processing["behavior"]["unit_labels"].data[:].shape[0]
        tw = 2

    def _extract_quality_metrics(self, nwb):
        metrics = {}
        for metric_name in METRIC_NAMES.values():
            metrics[metric_name] = nwb.processing["behavior"][f"metric-{metric_name}"].data[:]
        return metrics

    def spikes(self):
        return self.nwb.processing["behavior"]["trial_spike_times"].data[:]  # (units, trials, 700)

    def trial_motion_directions(self):
        return self.nwb.processing["behavior"]["trial_motion_directions"].data[:]

    def trial_block_idxs(self):
        return self.nwb.processing["behavior"]["trial_block_idx"].data[:]

    def trial_durations(self):
        return self.nwb.processing["behavior"]["trial_spike_duration_idxs"].data[:]  # (trials, 2)  [start, stop]

    def probe_units(self):
        return self.units()[:, self.probe_trial_idxs]

    def saccade_units(self):
        return self.units()[:, self.saccade_trial_idxs]

    def mixed_units(self):
        return self.units()[:, self.mixed_trial_idxs]

    def rp_peri_units(self):
        if self.use_normalized_units:
            return self.nwb.processing["behavior"]["normalized_trial_rp_peri_response_firing_rates"].data[:]
        else:
            return self.nwb.processing["behavior"]["trial_rp_peri_response_firing_rates"].data[:]

    def units(self):
        if self.use_normalized_units:
            return self.nwb.processing["behavior"]["normalized_trial_response_firing_rates"].data[:]
        else:
            return self.nwb.processing["behavior"]["trial_response_firing_rates"].data[:]  # units x trials x t

    def unit_filter_premade(self) -> UnitFilter:
        return self.unit_filter_qm().append(
            self.unit_filter_probe_zeta().append(self.unit_filter_custom(5, .2, 1, 1, .9, .4))
        )

    def unit_filter_qm(self) -> UnitFilter:
        return QualityMetricsUnitFilter(self.quality_metrics, self.num_units)
    
    def unit_filter_probe_zeta(self) -> UnitFilter:
        return ZetaUnitFilter(self.nwb.processing["behavior"]["probe_zeta_scores"].data[:])

    def unit_filter_custom(self, spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore) -> UnitFilter:
        return CustomUnitFilter(
            spike_count_threshold,
            trial_threshold,
            missing_threshold,
            min_missing,
            baseline_mean_zscore,
            baseline_time_std_zscore,
            self.nwb.processing["behavior"]["trial_spike_times"].data[:],
            self.units(),
            self.probe_trial_idxs,
            self.num_units
        )

    def trial_motion_filter(self, motion_direction) -> TrialFilter:
        return MotionDirectionTrialFilter(motion_direction, self.trial_motion_directions())

    def trial_filter_rp_peri(self, additional_filters):
        return RelativeTrialFilter(additional_filters, self.mixed_trial_idxs)

    def trial_filter_rp_extra(self):
        return BasicFilter(self.probe_trial_idxs, self.num_trials)

    def trial_filter_rs(self):
        return BasicFilter(self.saccade_trial_idxs, self.num_trials)

    def trial_filter_rmixed(self):
        return BasicFilter(self.mixed_trial_idxs, self.num_trials)

