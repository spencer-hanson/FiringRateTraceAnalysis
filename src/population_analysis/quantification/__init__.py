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
        actual = self.quan.calculate(self.class_1_data, self.class_2_data)
        print(f"Actual value {actual}")

        all_values = np.vstack([self.class_1_data, self.class_2_data])
        values_len = len(all_values)
        half = int(values_len / 2)
        quan_values = []

        print("Calculating quantification 10k times -", end="")
        total = 10000
        one_tenth = int(total / 10)
        for progress in range(total):  # Calculate quan 10k times
            if progress % one_tenth == 0:
                print(f" {round(progress/total, 2)*100}%", end="")

            np.random.shuffle(all_values)
            new_class_1 = all_values[:half]
            new_class_2 = all_values[half:]
            val = self.quan.calculate(new_class_1, new_class_2)
            quan_values.append(val)
        return quan_values
