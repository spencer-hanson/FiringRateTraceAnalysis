import numpy as np
import matplotlib.pyplot as plt
from population_analysis.quantification import Quantification


class EuclidianQuantification(Quantification):
    def calculate(self, class_1_data, class_2_data):
        mean_1 = np.average(class_1_data, axis=0)
        mean_2 = np.average(class_2_data, axis=0)

        # plt.plot(range(len(mean_1)), mean_1, color="red")
        # plt.plot(range(len(mean_2)), mean_2, color="blue")
        # plt.show()
        # tw = 2
        dist = np.linalg.norm(mean_1-mean_2)
        return dist
