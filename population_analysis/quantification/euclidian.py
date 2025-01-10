import numpy as np
import matplotlib.pyplot as plt
from population_analysis.quantification import Quantification


class EuclidianQuantification(Quantification):

    def __init__(self, name=None):
        if name is None:
            name = "Euclidian"
        super().__init__(name)

    def calculate(self, class_1_data, class_2_data):
        assert len(class_1_data.shape) == 2
        assert len(class_2_data.shape) == 2

        mean_1 = np.average(class_1_data, axis=1)  # expects shape to be (units, trials) pass in each time sep
        mean_2 = np.average(class_2_data, axis=1)  # Averaging over trials

        dist = np.linalg.norm(mean_1-mean_2)
        return dist
