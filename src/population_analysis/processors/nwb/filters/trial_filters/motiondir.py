import numpy as np

from population_analysis.processors.nwb.filters.trial_filters import TrialFilter


class MotionDirectionTrialFilter(TrialFilter):
    def __init__(self, motion_direction, trial_directions):
        self.motdir = motion_direction
        self.trial_directions = trial_directions

        super().__init__(self._get_passing_func(), len(self.trial_directions))

    def _get_passing_func(self):
        passing_idxs = np.where(self.trial_directions == self.motdir)[0]

        def passing(trial_num):
            if trial_num in passing_idxs:
                return True
            return False

        return passing

    def get_basename(self):
        return "motiondir"
