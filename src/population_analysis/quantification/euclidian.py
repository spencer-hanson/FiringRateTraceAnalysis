import numpy as np

from population_analysis.quantification import Quantification


class EuclidianQuantification(Quantification):
    def calculate(self, class_1_data, class_2_data):
        mean_1 = np.average(class_1_data)
        mean_2 = np.average(class_2_data)

        dist = np.linalg.norm(mean_1-mean_2)
        return dist
