from population_analysis.consts import METRIC_THRESHOLDS
from population_analysis.processors.nwb.filters.unit_filters import UnitFilter


class QualityMetricsUnitFilter(UnitFilter):
    def __init__(self, quality_metrics, num_units):
        self.quality_metrics = quality_metrics

        super().__init__(self.get_passfunc(), num_units)

    def get_passfunc(self):
        def passing_quality_metrics(unit_num) -> bool:
            """
            My approach for handling the manually curated spikes is to pass everything through a 2-part filter.
            For the first part, I ask: What is the quality label? If it's 1 I let the unit through, if it's 0,
            I move on to step 2. For the second part, I ask: Does the unit meet spike-sorting quality metric thresholds.
            If yes, I let it through. If no, I exclude it.
            """
            label = self.quality_metrics["quality_labeling"][unit_num]
            if label == 1:
                return True

            for metric_name, metric_func in METRIC_THRESHOLDS.items():
                val = self.quality_metrics[metric_name][unit_num]
                result = metric_func(val)
                if not result:
                    return False

            return True

        return passing_quality_metrics

    def get_basename(self):
        return "qm"

