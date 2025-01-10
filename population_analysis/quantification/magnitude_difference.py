import numpy as np

from population_analysis.quantification import Quantification


class MagDiffQuantification(Quantification):
    def __init__(self, name=None):
        if name is None:
            name = "MagnitudeDifference"
        super().__init__(name)

    def calculate(self, class_1_data, class_2_data) -> float:
        # expects shape to be (units, trials) pass in each time sep
        # expects shape to be (units, trials) pass in each time sep
        mean_1 = np.mean(class_1_data, axis=1)
        mean_2 = np.mean(class_2_data, axis=1)

        mag1 = np.linalg.norm(mean_1)
        mag2 = np.linalg.norm(mean_2)

        return mag2 - mag1

    def get_name(self):
        return self.name

