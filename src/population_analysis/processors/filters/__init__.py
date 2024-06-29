import numpy as np


class Filter(object):
    def __init__(self, func, num_units):
        self._funcs = [func]  # list of funcs like func(absolute_unit_num) -> bool passes filter
        self._idxs = None
        self.num_units = num_units
        self.names = [self.get_basename()]

    def get_basename(self):
        return "empty"

    def copy(self):
        f = Filter(self._funcs[0], self.num_units)
        for func in self._funcs[1:]:
            f._funcs.append(func)
        f.names = self.names
        return f

    def get_name(self):
        return "_".join(self.names)

    @staticmethod
    def empty(num_units):
        return Filter(lambda v: True, num_units)

    def idxs(self):
        # return a list of indexes into the full list that pass the filter
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

    def append(self, filt: 'Filter') -> 'Filter':
        if filt.num_units != self.num_units:
            raise ValueError(
                f"Cannot append filter, unit nums don't match! self != other {self.num_units} != {filt.num_units}")

        self._funcs.extend(filt._funcs)
        self.names.append(filt.get_basename())
        return self


class BasicFilter(Filter):
    def __init__(self, els, num_elements):
        self.els = els

        super().__init__(self._get_passing_func(), num_elements)

    def _get_passing_func(self):
        def passing(num):
            if num in self.els:
                return True
            return False

        return passing

    def get_basename(self):
        return "basic"
