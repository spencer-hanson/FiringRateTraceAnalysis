import numpy as np

from population_analysis.processors.filters.trial_filters import TrialFilter


class ProbeOffsetTrialFilter(TrialFilter):
    def __init__(self, mixed_rel_timestamps, bins, bin_num):
        self.bin_idxs = np.digitize(mixed_rel_timestamps, bins)
        self.bin_num = bin_num

        super().__init__(self._get_passing_func(), len(mixed_rel_timestamps))

    def _get_passing_func(self):
        passing_idxs = np.where(self.bin_idxs == self.bin_num)[0]

        def passing(trial_num):
            if trial_num in passing_idxs:
                return True
            return False

        return passing

    def get_basename(self):
        return "probeoffset"
