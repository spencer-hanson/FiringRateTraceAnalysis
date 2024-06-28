import numpy as np

from population_analysis.processors.experiments.saccadic_modulation import ModulationTrialGroup


class FiringRateCalculator(object):
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

    def calculate(self):
        # |--A-10sec---|--B-10sec---|-C-.2sec--|---Probe--|
        # baseline mean C
        # std over just A

        import matplotlib.pyplot as plt

        def p(d):
            plt.plot(d)
            plt.show()

        print("Calculating firing rates..", end="")
        unit_std_groups = {i: [] for i in range(self.firing_rates.shape[0])}  # {unit_num: [<baseline mean1>, ..], ..}
        normalized_frs = np.full((self.num_units, self.num_trials, 35), -888)

        all_trial_firing_rates = []
        all_normalized_firing_rates = []

        for trial_idx, trial in enumerate(self.trial_group.all_trials()):
            if trial_idx % int(self.num_trials/10) == 0:
                print(f" {round(trial_idx / self.num_trials, 2)*100}%", end="")

            trial_start_idx = trial.start_idx
            trial_end_idx = trial.end_idx

            trial_firing_rates = []
            trial_normalized_firing_rates = []

            for unit_num in range(self.num_units):
                v = self.firing_rates[unit_num, trial_start_idx:trial_end_idx]
                trial_firing_rates.append(v)
                baseline = self.firing_rates[unit_num, trial_start_idx:trial_start_idx + 10]  # Mean firing rate from -200, 0ms (relative to probe)
                baseline = v - np.mean(baseline)
                trial_normalized_firing_rates.append(baseline)  # Will be adding on firingrates
                tw = 2

            all_normalized_firing_rates.append(trial_normalized_firing_rates)
            all_trial_firing_rates.append(trial_firing_rates)

        all_normalized_firing_rates = np.array(all_normalized_firing_rates)
        all_trial_firing_rates = np.array(all_trial_firing_rates)

        print("\nNormalizing firing rates..")
        preferred = self._calculate_preferred_motion_direction(all_trial_firing_rates)  # (units,)
        # Grab std baselines
        for unit_num in range(self.num_units):
            motdir = preferred[unit_num]

            for trial_idx, trial in enumerate(self.trial_group.get_trials_by_motion(motdir)):
                unit_std_groups[unit_num].append(self.firing_rates[unit_num, trial.start_idx - 1000:trial.start_idx - 500])  # from -1000idx, -500idx is -20sec, -10sec

        for unit_num in range(self.num_units):
            ustd = np.std(unit_std_groups[unit_num])
            tw = 2

