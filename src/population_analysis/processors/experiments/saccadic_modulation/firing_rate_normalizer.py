import os

import numpy as np

from population_analysis.processors.experiments.saccadic_modulation import ModulationTrialGroup
from population_analysis.processors.experiments.saccadic_modulation.rp_peri_calculator import RpPeriCalculator


class FiringRateNormalizer(object):
    FIRING_RATE_FILENAME = "calc_firingrates.npy"
    NORMALIZED_FILENAME = "calc_norm_firingrates.npy"
    RP_PERI_FIRING_RATE = "calc_rpperi_firingrates.npy"
    RP_PERI_NORMALIZED = "calc_rpperi_norm_firingrates.npy"
    LARGE_NORMALIZED = "calc_large_norm_firingrates.npy"
    BASELINE_IDXS = [0, 10]  # Baseline is 0:10

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
        rppc = RpPeriCalculator(firing_rates, self.trial_group.trial_type_idxs["saccade"], self.trial_group.trial_type_idxs["mixed"], self.trial_group)
        raw_rpp = rppc.calculate()
        return raw_rpp

    def _check_for_nan(self, arr):
        if bool(np.where(np.isnan(arr.ravel()))[0]):
            raise ValueError("Array contains a NaN! Fix pls!")
        tw = 2

    def _load_firingrates(self):
        return {
            "firing_rate": np.load(FiringRateNormalizer.FIRING_RATE_FILENAME, mmap_mode='r'),
            "normalized_firing_rate": np.load(FiringRateNormalizer.NORMALIZED_FILENAME, mmap_mode='r'),
            "rp_peri_firing_rate": np.load(FiringRateNormalizer.RP_PERI_FIRING_RATE, mmap_mode='r'),
            "rp_peri_normalized_firing_rate": np.load(FiringRateNormalizer.RP_PERI_NORMALIZED, mmap_mode='r'),
            "largerange_normalized_firing_rate": np.load(FiringRateNormalizer.LARGE_NORMALIZED, mmap_mode='r')
        }

    def _subtract_baseline_rp_peri(self, rp_peri):
        # rp_peri is (units, trials, t)
        # start and end are the indexes into the arr where the baseline mean should be taken
        baseline = np.mean(rp_peri[:, :, FiringRateNormalizer.BASELINE_IDXS[0]:FiringRateNormalizer.BASELINE_IDXS[1]], axis=2)  # Baseline is the first 200ms before the probe
        normalized = rp_peri - baseline[:, :, None]
        return normalized

    def calculate(self, load_precalculated):
        # |--A-10sec---|--B-10sec---|-C-.2sec--|---Probe--|
        # baseline mean C
        # std over just A

        special_firing_rates = [
            ("firing_rate", FiringRateNormalizer.FIRING_RATE_FILENAME),
            ("normalized_firing_rate", FiringRateNormalizer.NORMALIZED_FILENAME),
            ("rp_peri_firing_rate", FiringRateNormalizer.RP_PERI_FIRING_RATE),
            ("rp_peri_normalized_firing_rate", FiringRateNormalizer.RP_PERI_NORMALIZED),
            ("largerange_normalized_firing_rate", FiringRateNormalizer.LARGE_NORMALIZED)
        ]

        if load_precalculated:
            print("Attempting to load a precalculated firing rate from local directory..")
            all_cached = True
            for nm, filen in special_firing_rates:
                if not os.path.exists(filen):
                    print(f"Precalculated file '{filen}' doesn't exist, generating..")
                    all_cached = False
            if all_cached:
                return self._load_firingrates()

        print("Normalizing trial firing rates..", end="")
        unit_std_groups = {i: [] for i in range(self.firing_rates.shape[0])}  # {unit_num: [<baseline mean1>, ..], ..}

        all_trial_firing_rates = []  # (trials, units, t) see rpperi but with diff idxs
        all_normalized_firing_rates = []
        all_rp_peri_firing_rates = []  # Keep track of rp_peri calculations arr like [[[firing rates of -1000idx, end_idx for one unit], <another unit>], ..more trials]
        all_largerange_normalized_firing_rates = []

        first_trial = self.trial_group.all_trials()[0]
        trial_idx_len = first_trial.end_idx - first_trial.start_idx

        for trial_idx, trial in enumerate(self.trial_group.all_trials()):
            if trial_idx % int(self.num_trials/10) == 0:
                print(f" {round(trial_idx / self.num_trials, 2)*100}%", end="")

            trial_start_idx = trial.start_idx
            trial_end_idx = trial.end_idx
            assert trial_end_idx - trial_start_idx == trial_idx_len  # Make sure each trial is the same length
            assert trial_end_idx < self.firing_rates.shape[1]  # Need to have enough firing rate data for all trials
            assert trial_start_idx >= 0

            trial_firing_rates = []
            trial_normalized_firing_rates = []
            rp_peri_firing_rates = []
            largerange_normalized_firing_rates = []  # Large range of firing rates for misc calcs, from [-1400, +1400] rel. to event time

            for unit_num in range(self.num_units):
                # regular firing rates
                response = self.firing_rates[unit_num, trial_start_idx:trial_end_idx]
                assert len(response) > 0
                trial_firing_rates.append(response)

                # normalized firing rates
                baseline = self.firing_rates[unit_num, trial_start_idx + FiringRateNormalizer.BASELINE_IDXS[0]:trial_start_idx + FiringRateNormalizer.BASELINE_IDXS[1]]  # Mean firing rate from -200, 0ms (relative to probe)
                assert len(baseline) > 0
                baseline = np.mean(baseline)
                trial_normalized_firing_rates.append(response - baseline)  # Will be adding on firingrates

                # rp peri firing rates
                # Include whole range so we can calc all stats
                rp_peri_firing_rates.append(
                    self.firing_rates[unit_num, trial_start_idx - 35: trial_end_idx + 35]
                )

                largerange_normalized_firing_rates.append(
                    self.firing_rates[unit_num, trial_start_idx - 35: trial_end_idx + 35] - baseline
                )

            all_trial_firing_rates.append(trial_firing_rates)
            all_normalized_firing_rates.append(trial_normalized_firing_rates)
            all_rp_peri_firing_rates.append(rp_peri_firing_rates)
            all_largerange_normalized_firing_rates.append(largerange_normalized_firing_rates)

        print("")
        all_normalized_firing_rates = np.array(all_normalized_firing_rates).swapaxes(0, 1)
        all_trial_firing_rates = np.array(all_trial_firing_rates).swapaxes(0, 1)
        all_rp_peri_firing_rates = np.array(all_rp_peri_firing_rates).swapaxes(0, 1)
        all_largerange_normalized_firing_rates = np.array(all_largerange_normalized_firing_rates).swapaxes(0, 1)

        self._check_for_nan(all_normalized_firing_rates)
        self._check_for_nan(all_trial_firing_rates)
        self._check_for_nan(all_rp_peri_firing_rates)
        self._check_for_nan(all_largerange_normalized_firing_rates)

        preferred = self._calculate_preferred_motion_direction(all_trial_firing_rates)  # (units,)

        print("Saving non-std'd firing rates to file..")
        np.save(FiringRateNormalizer.FIRING_RATE_FILENAME, all_trial_firing_rates)
        firing_rate_baselines = np.mean(all_trial_firing_rates[:, :, 0:10], axis=2)
        firing_rate_baselines = np.mean(firing_rate_baselines, axis=1)[:, None, None]  # Average over trials and set up to be broadcast onto rp_peri
        del all_trial_firing_rates

        print("Calculating RpPeri..")
        all_rp_peri_firing_rates = self._calculate_rp_peri(all_rp_peri_firing_rates)  # comes out as (units, trials, t)
        print("Saving RpPeri firing rates to file..")
        np.save(FiringRateNormalizer.RP_PERI_FIRING_RATE, all_rp_peri_firing_rates)

        print("Normalizing all firing rates..")
        # Grab std baselines
        for unit_num in range(self.num_units):
            motdir = preferred[unit_num]

            for trial_idx, trial in enumerate(self.trial_group.get_trials_by_motion(motdir)):
                arrr = self.firing_rates[unit_num, trial.start_idx - 1000:trial.start_idx - 500]
                baseline_frs = np.mean(arrr)
                assert len(arrr) > 0
                unit_std_groups[unit_num].append(baseline_frs)  # from -1000idx, -500idx is -20sec, -10sec

        raw_unit_stds = []
        for unit_num in range(self.num_units):
            ustd = np.std(unit_std_groups[unit_num])
            raw_unit_stds.append(ustd)

        raw_unit_stds = np.array(raw_unit_stds)
        raw_unit_stds[raw_unit_stds == 0] = 1
        unit_stds = np.broadcast_to(raw_unit_stds[:, None, None], (self.num_units, self.num_trials, 35))
        large_unit_stds = np.broadcast_to(raw_unit_stds[:, None, None], (self.num_units, self.num_trials, 35*3))

        all_normalized_firing_rates /= unit_stds
        all_largerange_normalized_firing_rates /= large_unit_stds

        all_normalized_rp_peri_firing_rates = all_rp_peri_firing_rates - firing_rate_baselines
        all_normalized_rp_peri_firing_rates /= np.mean(unit_stds, axis=1)[:, None, :]
        # Re-normalize the baseline since the subtracted RpPeri will have a different baseline (after zscoring from above)
        all_normalized_rp_peri_firing_rates = self._subtract_baseline_rp_peri(all_normalized_rp_peri_firing_rates)

        print("Saving normalized & normalized RpPeri firing rates to file..")
        np.save(FiringRateNormalizer.NORMALIZED_FILENAME, all_normalized_firing_rates)
        np.save(FiringRateNormalizer.RP_PERI_NORMALIZED, all_normalized_rp_peri_firing_rates)
        np.save(FiringRateNormalizer.LARGE_NORMALIZED, all_largerange_normalized_firing_rates)

        del all_normalized_rp_peri_firing_rates
        del all_rp_peri_firing_rates
        del all_normalized_firing_rates
        del all_largerange_normalized_firing_rates

        return self._load_firingrates()
