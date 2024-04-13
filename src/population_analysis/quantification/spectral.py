from population_analysis.quantification import Quantification
import numpy as np


class SpectralQuantification(Quantification):
    def _decompose(self, dataset):
        # dataset is (trials, units)
        hist_data = []
        for i in range(dataset.shape[0]):
            hist = np.histogram(dataset[i], bins=np.arange(0, 1, .01))
            hist_data.append(hist[0])
        hist_data = np.array(hist_data)
        avgd = np.mean(hist_data, axis=0)
        # import matplotlib.pyplot as plt
        # plt.plot(avgd)
        # plt.show()
        tw = 2
        return avgd

    def calculate(self, class_1_data, class_2_data):
        c1 = self._decompose(class_1_data)
        c2 = self._decompose(class_2_data)
        dist = np.linalg.norm(c1-c2)
        tw = 2
        return dist

