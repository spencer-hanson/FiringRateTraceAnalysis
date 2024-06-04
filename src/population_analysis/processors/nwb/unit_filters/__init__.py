import numpy as np


class UnitFilter(object):
    def __init__(self, func, num_units):
        self._funcs = [func]  # list of funcs like func(absolute_unit_num) -> bool passes filter
        self._idxs = None
        self.num_units = num_units
        self.names = [self.get_basename()]

    def get_basename(self):
        return "empty"

    def get_name(self):
        return "_".join(self.names)

    @staticmethod
    def empty(num_units):
        return UnitFilter(lambda v: True, num_units)

    def idxs(self):
        # return a list of indexes into the full list of units that pass the filter
        if self._idxs is None:
            passing = []
            for num in range(self.num_units):
                if self.passes_abs(num):
                    passing.append(num)
            self._idxs = np.array(passing)

        return self._idxs

    def len(self):
        # return the length of the indexes
        return len(self.idxs())

    def passes_abs(self, abs_unit_num) -> bool:
        for func in self._funcs:
            if not func(abs_unit_num):
                return False
        return True

    def append(self, unit_filter: 'UnitFilter') -> 'UnitFilter':
        if unit_filter.num_units != self.num_units:
            raise ValueError(
                f"Cannot append unit filter, unit nums don't match! self != other {self.num_units} != {unit_filter.num_units}")

        self._funcs.extend(unit_filter._funcs)
        self.names.append(unit_filter.get_basename())
        return self
