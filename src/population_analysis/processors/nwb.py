import os

import numpy as np
from pynwb import NWBHDF5IO


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
            mixed_filtered_idxs = np.abs(self.mixed_rel_timestamps) <= 0.02  # 20 ms
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