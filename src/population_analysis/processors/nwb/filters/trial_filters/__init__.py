from population_analysis.processors.nwb.filters import Filter


class TrialFilter(Filter):
    def __init__(self, passing_func, num_trials):
        super().__init__(passing_func, num_trials)
