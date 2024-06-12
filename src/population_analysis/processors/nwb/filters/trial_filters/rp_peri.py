import numpy as np

from population_analysis.processors.nwb.filters.trial_filters import TrialFilter


class RpPeriTrialFilter(TrialFilter):
    def __init__(self, regular_filter: TrialFilter, abs_idx_mappings, rp_peri):
        self.regular_filter = regular_filter
        # list of indexes that correspond 1-1 with rp_peri, where the first element's value is the
        # index into the global trials of the first rp_peri element
        # and the elements in the mappings represent their corresponding value in rp_peri
        # so [<mixed trial1>, ...] <-> <rp_peri arr>
        self.abs_idx_mappings = abs_idx_mappings
        self.rp_peri = rp_peri

        super().__init__(self._get_passing_func(), len(rp_peri))

    def _get_passing_func(self):
        def passing(rp_peri_trial_num):
            trial_num = self.abs_idx_mappings[rp_peri_trial_num]
            if trial_num in self.regular_filter.idxs():
                return True
            return False

        return passing

    def get_basename(self):
        return "rpperi"
