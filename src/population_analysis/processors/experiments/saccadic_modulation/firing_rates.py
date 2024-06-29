import os

import numpy as np

from population_analysis.processors.experiments.saccadic_modulation import ModulationTrialGroup
from population_analysis.processors.experiments.saccadic_modulation.rp_peri_calculator import RpPeriCalculator


class FiringRateCalculator(object):
    FIRING_RATE_FILENAME = "calc_firingrates.npy"
    NORMALIZED_FILENAME = "calc_norm_firingrates.npy"
    RP_PERI_FIRING_RATE = "calc_rpperi_firingrates.npy"
    RP_PERI_NORMALIZED = "calc_rpperi_norm_firingrates.npy"

    def __init__(self, firing_rates, trial_group: ModulationTrialGroup):
        self.firing_rates = firing_rates
        self.trial_group = trial_group
        self.num_units = self.firing_rates.shape[0]
        self.num_trials = self.trial_group.num_trials

    def _calculate_preferred_motion_direction(self, trial_firing_rates):
        preferred = []  # (units,) arr with preferred motion dir, like [+-1, ..]
        for unit_num in range(self.num_units):
            # Use the value of the response between time 8-12idx (time of probe is 10) so
            # -2*20 = -40, 2*20 = 40, (-40ms, 40ms) window
            neg_idxs = self.trial_group.get_trial_idxs_by_motion(-1)
            pos_idxs = self.trial_group.get_trial_idxs_by_motion(1)

            neg_mean = np.abs(np.mean(np.mean(trial_firing_rates[unit_num][neg_idxs][:, 8:12], axis=0), axis=0))  # average over the trial event responses
            pos_mean = np.abs(np.mean(np.mean(trial_firing_rates[unit_num][pos_idxs][:, 8:12], axis=0), axis=0))

            if neg_mean > pos_mean:
                preferred.append(-1)
            else:
                preferred.append(1)

        return np.array(preferred)

    def _calculate_rp_peri(self, firing_rates):
        # firing_rates (units, trials, t)
        rppc = RpPeriCalculator(firing_rates, self.trial_group.trial_type_idxs["saccade"], self.trial_group.trial_type_idxs["mixed"])
        raw_rpp = rppc.calculate()
        return raw_rpp

    def calculate(self, load_precalculated=True):
        # |--A-10sec---|--B-10sec---|-C-.2sec--|---Probe--|
        # baseline mean C
        # std over just A

        if load_precalculated:
            print("Attempting to load a precalculated firing rate from local directory..")
            if os.path.exists(FiringRateCalculator.FIRING_RATE_FILENAME) and os.path.exists(FiringRateCalculator.NORMALIZED_FILENAME) and os.path.exists(FiringRateCalculator.RP_PERI_FIRING_RATE) and os.path.exists(FiringRateCalculator.RP_PERI_NORMALIZED):
                return {
                    "firing_rate": np.load(FiringRateCalculator.FIRING_RATE_FILENAME, mmap_mode='r'),
                    "normalized_firing_rate": np.load(FiringRateCalculator.NORMALIZED_FILENAME, mmap_mode='r'),
                    "rp_peri_firing_rate": np.load(FiringRateCalculator.RP_PERI_FIRING_RATE, mmap_mode='r'),
                    "rp_peri_normalized_firing_rate": np.load(FiringRateCalculator.RP_PERI_NORMALIZED, mmap_mode='r')
                }
            else:
                print(f"One of the precalculated files for firing rate calculations does not exist, generating..")

        print("Calculating firing rates..", end="")
        unit_std_groups = {i: [] for i in range(self.firing_rates.shape[0])}  # {unit_num: [<baseline mean1>, ..], ..}

        all_trial_firing_rates = []  # (trials, units, t) see rpperi but with diff idxs
        all_normalized_firing_rates = []
        all_rp_peri_firing_rates = []  # Keep track of rp_peri calculations arr like [[[firing rates of -1000idx, end_idx for one unit], <another unit>], ..more trials]
        all_normalized_rp_peri_firing_rates = []

        for trial_idx, trial in enumerate(self.trial_group.all_trials()):
            if trial_idx % int(self.num_trials/10) == 0:
                print(f" {round(trial_idx / self.num_trials, 2)*100}%", end="")

            trial_start_idx = trial.start_idx
            trial_end_idx = trial.end_idx

            trial_firing_rates = []
            trial_normalized_firing_rates = []
            rp_peri_firing_rates = []
            normalized_rp_peri_firing_rates = []

            for unit_num in range(self.num_units):
                # regular firing rates
                v = self.firing_rates[unit_num, trial_start_idx:trial_end_idx]
                trial_firing_rates.append(v)

                # normalized firing rates
                baseline = self.firing_rates[unit_num, trial_start_idx:trial_start_idx + 10]  # Mean firing rate from -200, 0ms (relative to probe)
                baseline = np.mean(baseline)
                trial_normalized_firing_rates.append(v - baseline)  # Will be adding on firingrates

                # rp peri firing rates
                # Include whole range so we can calc all stats
                rp_peri_firing_rates.append(
                    self.firing_rates[unit_num, trial_start_idx: trial_end_idx]
                )
                normalized_rp_peri_firing_rates.append(
                    self.firing_rates[unit_num, trial_start_idx: trial_end_idx] - baseline
                )

            all_trial_firing_rates.append(trial_firing_rates)
            all_normalized_firing_rates.append(trial_normalized_firing_rates)
            all_rp_peri_firing_rates.append(rp_peri_firing_rates)
            all_normalized_rp_peri_firing_rates.append(normalized_rp_peri_firing_rates)

        all_normalized_firing_rates = np.array(all_normalized_firing_rates).swapaxes(0, 1)
        all_trial_firing_rates = np.array(all_trial_firing_rates).swapaxes(0, 1)
        all_rp_peri_firing_rates = np.array(all_rp_peri_firing_rates).swapaxes(0, 1)
        all_normalized_rp_peri_firing_rates = np.array(all_normalized_rp_peri_firing_rates).swapaxes(0, 1)

        preferred = self._calculate_preferred_motion_direction(all_trial_firing_rates)  # (units,)

        print("Saving firing rates to file..")
        np.save(FiringRateCalculator.FIRING_RATE_FILENAME, all_trial_firing_rates)
        del all_trial_firing_rates

        print("Calculating RpPeri..")
        all_rp_peri_firing_rates = self._calculate_rp_peri(all_rp_peri_firing_rates)  # (units, trials, t)

        print("Saving RpPeri firing rates to file..")
        np.save(FiringRateCalculator.RP_PERI_FIRING_RATE, all_rp_peri_firing_rates)
        del all_rp_peri_firing_rates

        print("Calculating normalized RpPeri..")
        all_normalized_rp_peri_firing_rates = self._calculate_rp_peri(all_normalized_rp_peri_firing_rates)  # has all trials, need to filter out only mixed

        print("\nNormalizing all firing rates..")

        # Grab std baselines
        for unit_num in range(self.num_units):
            motdir = preferred[unit_num]

            for trial_idx, trial in enumerate(self.trial_group.get_trials_by_motion(motdir)):
                baseline_frs = np.mean(self.firing_rates[unit_num, trial.start_idx - 1000:trial.start_idx - 500])
                unit_std_groups[unit_num].append(baseline_frs)  # from -1000idx, -500idx is -20sec, -10sec

        raw_unit_stds = []
        for unit_num in range(self.num_units):
            ustd = np.std(unit_std_groups[unit_num])
            raw_unit_stds.append(ustd)

        raw_unit_stds = np.array(raw_unit_stds)
        raw_unit_stds[raw_unit_stds == 0] = 1
        unit_stds = np.broadcast_to(raw_unit_stds[:, None, None], (self.num_units, self.num_trials, 35))
        # Need a different shape for the rp_peri stds
        rpperi_unit_stds = np.broadcast_to(raw_unit_stds[:, None, None], (self.num_units, all_normalized_rp_peri_firing_rates.shape[1], 35))

        all_normalized_firing_rates /= unit_stds
        all_normalized_rp_peri_firing_rates /= rpperi_unit_stds

        print("Saving normalized & normalized RpPeri firing rates to file..")
        np.save(FiringRateCalculator.NORMALIZED_FILENAME, all_normalized_firing_rates)
        np.save(FiringRateCalculator.RP_PERI_NORMALIZED, all_normalized_rp_peri_firing_rates)

        del all_normalized_rp_peri_firing_rates
        del all_normalized_firing_rates

        return {
            "firing_rate": np.load(FiringRateCalculator.FIRING_RATE_FILENAME, mmap_mode='r'),
            "normalized_firing_rate": np.load(FiringRateCalculator.NORMALIZED_FILENAME, mmap_mode='r'),
            "rp_peri_firing_rate": np.load(FiringRateCalculator.RP_PERI_FIRING_RATE, mmap_mode='r'),
            "rp_peri_normalized_firing_rate": np.load(FiringRateCalculator.RP_PERI_NORMALIZED, mmap_mode='r')
        }
