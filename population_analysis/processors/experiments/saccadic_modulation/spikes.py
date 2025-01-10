import os

import numpy as np

from population_analysis.consts import SPIKE_BIN_MS
from population_analysis.processors.experiments.saccadic_modulation import ModulationTrialGroup


class SpikeTrialOrganizer(object):
    SPIKE_TRIALS_FILENAME = "calc_spike_trials.npy"

    def __init__(self, raw_spike_times, trialgroup: ModulationTrialGroup):
        self.all_spikes = raw_spike_times
        self.trialgroup = trialgroup

    def calculate(self, load_precalculated):
        if load_precalculated:
            print("Attempting to load a precalculated spike trials from local directory..")
            if os.path.exists(SpikeTrialOrganizer.SPIKE_TRIALS_FILENAME):
                return np.load(SpikeTrialOrganizer.SPIKE_TRIALS_FILENAME, mmap_mode='r')
            else:
                print(f"Precalculated file does not exist, generating..")

        all_spike_trials = []
        for tr in self.trialgroup.all_trials():
            units_spike_trial = []
            for unit_idx in range(self.all_spikes.shape[0]):
                val = self.all_spikes[unit_idx, tr.start_idx*SPIKE_BIN_MS:tr.end_idx*SPIKE_BIN_MS]  # spikes are binned in 20ms bins
                units_spike_trial.append(val)
            all_spike_trials.append(units_spike_trial)
        all_spike_trials = np.array(all_spike_trials).swapaxes(0, 1)  # want them in (units, trials, 700)

        with open(SpikeTrialOrganizer.SPIKE_TRIALS_FILENAME, "wb") as f:
            np.save(f, all_spike_trials)
        del all_spike_trials

        all_spike_trials = np.load(SpikeTrialOrganizer.SPIKE_TRIALS_FILENAME, mmap_mode='r')
        return all_spike_trials
