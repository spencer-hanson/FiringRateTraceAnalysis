import numpy as np
import random


class Quantification(object):
    def calculate(self, class_1_data, class_2_data):
        raise NotImplemented


class QuanDistribution(object):
    def __init__(self, class_1_data, class_2_data, quan: Quantification):
        self.quan: Quantification = quan
        self.class_1_data = class_1_data
        self.class_2_data = class_2_data

    def calculate(self):
        all_values = np.vstack(self.class_1_data, self.class_2_data)
        values_len = len(all_values)
        half = int(values_len / 2)
        quan_values = []

        for _ in range(10000):  # Calculate quan 10k times
            new_class_1 = np.empty((half, *self.class_1_data.shape[1:]))

            chosen_idxs = []
            count = 0
            for i in range(half):
                idx = random.randrange(values_len)
                chosen_idxs.append(idx)
                new_class_1[count, :] = all_values[count, :]
            chosen_idxs = set(chosen_idxs)
            not_chosen = set(range(values_len)).difference(chosen_idxs)

            new_class_2 = all_values[list(not_chosen)]

            val = self.quan.calculate(new_class_1, new_class_2)
            quan_values.append(val)
        return quan_values

