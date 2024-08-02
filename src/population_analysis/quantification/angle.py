import math

import numpy as np
import matplotlib.pyplot as plt
from population_analysis.quantification import Quantification


class AngleQuantification(Quantification):

    def __init__(self, name=None):
        if name is None:
            name = "Angle"
        super().__init__(name)

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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
        angle = self.angle_between(mean_1, mean_2)
        if abs(theta - angle) > 0.0001:
            raise ValueError("wut")

        return theta
