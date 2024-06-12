from population_analysis.processors.nwb.filters import Filter


class UnitFilter(Filter):
    pass

from .qm import QualityMetricsUnitFilter
from .zeta import ZetaUnitFilter
from .custom import CustomUnitFilter
