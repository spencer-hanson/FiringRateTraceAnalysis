import os

import numpy as np
from pynwb import NWBHDF5IO

from population_analysis.consts import TOTAL_TRIAL_MS


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

        # Filter out mixed trials that saccades are more than 20ms away from the probe
        self.mixed_rel_timestamps = nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]
        if filter_mixed:
            mixed_filtered_idxs = np.abs(self.mixed_rel_timestamps) <= 0.02  # only want mixed trials 20 ms within probe
            self.mixed_trial_idxs = self.mixed_trial_idxs[mixed_filtered_idxs]

        self.nwb = nwb
        tw = 2

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

    def filter_units_group(self, p, s, m, r, filter_idx):
        return p[filter_idx, :, :], s[filter_idx, :, :], m[filter_idx, :, :], r[filter_idx, :, :]

    def filter_units(self, filter_idx):
        p, s, m, r = self.unfiltered_units()
        return self.filter_units_group(p, s, m, r, filter_idx)

    def unfiltered_units(self):
        # (units, trials, t)
        probe_units = self.probe_units()
        saccade_units = self.saccade_units()
        mixed_units = self.mixed_units()
        rp_peri_units = self.rp_peri_units()
        return probe_units, saccade_units, mixed_units, rp_peri_units

    def zeta_units(self):
        probe_units, saccade_units, mixed_units, rp_peri_units = self.unfiltered_units()
        th = self.zeta_idxs()
        return probe_units[th, :, :], saccade_units[th, :, :], mixed_units[th, :, :], rp_peri_units[th, :, :]

    def zeta_idxs(self):
        return np.where(self.nwb.units["threshold_zeta_passes"])[0]

    def probe_zeta_idxs(self):
        return np.where(self.nwb.units["probe_zeta_scores"][:] < 0.01)[0]

    def activity_filtered_units_idxs(self, unit_filter=None):
        # unit_filter is an array of indexes to pre-filter the trial_spike_flags, return will be indexed relative
        bool_counts = self.nwb.units["trial_spike_flags"]  # units x trials x 700
        if unit_filter is not None:
            bool_counts = bool_counts[unit_filter, :, :]

        num_units = bool_counts.shape[0]
        passing_units_idxs = []

        for unit_num in range(num_units):
            unit_trials = bool_counts[unit_num, :, :]  # trials x 700
            trial_spike_sum = np.sum(unit_trials, axis=1)
            # want units that spike spike_count_threshold% of the time, in at least trial_threshold% of the trials

            spike_count_threshold = TOTAL_TRIAL_MS * .01
            trial_threshold = 0.2

            passing_trial_count = len(np.where(trial_spike_sum >= spike_count_threshold)[0])
            trial_count = bool_counts.shape[1]
            if trial_count * trial_threshold <= passing_trial_count:
                passing_units_idxs.append(unit_num)

        return passing_units_idxs

    def activity_filtered_units(self, unit_filter=None):
        if unit_filter is None:
            units = self.unfiltered_units()
        else:
            units = self.filter_units(unit_filter)
        idxs = self.activity_filtered_units_idxs(unit_filter)
        units = self.filter_units_group(*units, idxs)
        return units

