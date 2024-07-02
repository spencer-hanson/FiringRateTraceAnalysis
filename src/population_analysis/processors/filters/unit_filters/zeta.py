import numpy as np

from population_analysis.consts import UNIT_ZETA_P_VALUE
from population_analysis.processors.filters.unit_filters import UnitFilter


class ZetaUnitFilter(UnitFilter):
    def __init__(self, zeta_scores):
        num_units = len(zeta_scores)
        passing_idxs = np.where(zeta_scores < UNIT_ZETA_P_VALUE)[0]
        passing_units = np.array(list(range(num_units)))[passing_idxs]

        def does_pass(unit_num):
            p = unit_num in passing_units
            return p

        super().__init__(does_pass, num_units)

    def get_basename(self):
        return "zeta"
