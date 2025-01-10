import numpy as np
import matplotlib.pyplot as plt
from population_analysis.quantification import Quantification


class Euclidian2Quantification(Quantification):

    def __init__(self, name=None):
        if name is None:
            name = "Euclidian"
        super().__init__(name)
        self.data1s = []
        self.data2s = []

    def calculate(self, class_1_data, class_2_data):
        mean_1 = np.average(class_1_data, axis=1)  # expects shape to be (units, trials) pass in each time sep
        mean_2 = np.average(class_2_data, axis=1)  # Averaging over trials


        dist = np.linalg.norm(mean_1-mean_2)
        return dist
