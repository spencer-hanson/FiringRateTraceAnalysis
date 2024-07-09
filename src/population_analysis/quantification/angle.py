import math

import numpy as np
import matplotlib.pyplot as plt
from population_analysis.quantification import Quantification


class AngleQuantification(Quantification):

    def __init__(self, name=None):
        if name is None:
            name = "Angle"
        super().__init__(name)

    def calculate(self, class_1_data, class_2_data):
        assert len(class_1_data.shape) == 2
        assert len(class_2_data.shape) == 2

        mean_1 = np.average(class_1_data, axis=1)  # expects shape to be (units, trials) pass in each time sep
        mean_2 = np.average(class_2_data, axis=1)  # Averaging over trials

        mag1 = np.linalg.norm(mean_1)
        mag2 = np.linalg.norm(mean_2)
        dot = np.dot(mean_1, mean_2)

        cos_th = dot / (mag1 * mag2)
        theta = math.acos(cos_th)

        return theta
