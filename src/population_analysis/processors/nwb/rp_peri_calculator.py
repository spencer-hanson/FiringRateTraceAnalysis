import numpy as np


class RpPeriCalculator(object):
    def __init__(self, firing_rates, saccade_idxs, mixed_idxs):
        self.fr = firing_rates
        self.sac_idxs = saccade_idxs
        self.mix_idxs = mixed_idxs

    def calculate(self):
        saccade_unit_trial_waveforms = self.fr[self.sac_idxs]  # (units, trials, t))

        # average is now units x t
        saccade_unit_average_waveforms = np.average(saccade_unit_trial_waveforms, axis=1)  # Average over saccade trials for each unit
        # Get mixed waveform unit-trials
        mixed_unit_trial_waveforms = self.fr[self.mix_idxs]  # (units, trials, t)

        # Reshape saccade_unit_average_waveforms to (units, 1, t) and broadcast against (units, trials, t)
        # to subtract saccade avg from all units in all trials
        saccade_unit_average_waveforms = saccade_unit_average_waveforms[:, None].swapaxes(-1,-2)
        mixed_peri_waveforms = mixed_unit_trial_waveforms - saccade_unit_average_waveforms

        # Clamp negative values, cannot have a negative firing rate
        mixed_peri_waveforms = np.clip(mixed_peri_waveforms, 0, None)

        return mixed_peri_waveforms
