import numpy as np

from population_analysis.processors.nwb.filters.trial_filters import TrialFilter


class BasicTrialFilter(TrialFilter):
    def __init__(self, trial_idxs, num_trials):
        self.trial_idxs = trial_idxs

        super().__init__(self._get_passing_func(), num_trials)

    def _get_passing_func(self):
        def passing(trial_num):
            if trial_num in self.trial_idxs:
                return True
            return False

        return passing

    def get_basename(self):
        return "basic"
