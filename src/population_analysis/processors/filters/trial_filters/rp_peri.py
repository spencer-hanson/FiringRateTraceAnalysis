from population_analysis.processors.filters.trial_filters import TrialFilter


class RelativeTrialFilter(TrialFilter):
    def __init__(self, regular_filter: TrialFilter, abs_idx_mappings):
        self.regular_filter = regular_filter
        # list of indexes that correspond 1-1 with arr, where the first element's value is the
        # index into the global trials of the first arr element
        # and the elements in the mappings represent their corresponding value in arr
        # so [<abs idx>, ...] <-> <arr idx>
        self.abs_idx_mappings = abs_idx_mappings
        super().__init__(self._get_passing_func(), len(abs_idx_mappings))

    def _get_passing_func(self):
        def passing(rel_trial_num):
            trial_num = self.abs_idx_mappings[rel_trial_num]
            if trial_num in self.regular_filter.idxs():
                return True
            return False

        return passing

    def get_basename(self):
        return "relative"
