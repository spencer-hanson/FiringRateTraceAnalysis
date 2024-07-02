import numpy as np
import matplotlib.pyplot as plt
from population_analysis.quantification import Quantification


class EuclidianQuantification(Quantification):

    def __init__(self, name=None):
        if name is None:
            name = "Euclidian"
        super().__init__(name)

    # def ani(self):
    #
    #     def create_ani(dataset1, dataset2, filename):
    #         plots = {"d": []}
    #         # copy from here
    #         fig = plt.figure()
    #         ax = fig.add_subplot()
    #         plt.clf()
    #
    #         def update(frame):
    #             idx = frame % 35
    #             [p.remove() for p in plots["d"]]
    #             # to_plot = [ax.plot(self.data1s[idx])[0], ax.plot(self.data2s[idx])[0]]
    #             # to_plot = [ax.plot(self.data1s[idx]-self.data2s[idx])[0]]
    #             to_plot = [plt.plot((dataset1 - dataset2)[idx, :])[0]]
    #
    #             ax.set_ylim(np.min(dataset1 - dataset2), np.max(dataset1 - dataset2))
    #             fig.suptitle(f"Time {idx} dist {np.linalg.norm(dataset1[idx] - dataset2[idx])}")
    #             plots["d"] = to_plot
    #             return to_plot
    #
    #         print("Starting euc plotting")
    #         ax.set_ylim(np.min(dataset1 - dataset2), np.max(dataset1 - dataset2))
    #         from matplotlib import animation
    #         ani = animation.FuncAnimation(fig=fig, func=update, frames=35)
    #         ani.save(filename=filename, writer="pillow")
    #         plots["d"] = []
    #         print(f"Done writing gif '{filename}'")
    #
    #
    #     def diff_over_time(dataset1, dataset2, title):
    #         diff_overt = dataset1 - dataset2
    #         plt.clf()
    #         fig = plt.figure()
    #         ax = fig.add_subplot()
    #
    #         [ax.plot(diff_overt[:, i]) for i in range(dataset1.shape[1])]
    #         fig.suptitle(title)
    #         plt.show()
    #
    #     d1 = np.array(self.data1s)
    #     d2 = np.array(self.data2s)
    #     # Diff v time for all units on same plot
    #     diff_over_time(d1, d2, "Original")
    #
    #     # Plot original diff gif
    #     plt.clf()
    #     dists_o = [np.linalg.norm(d1[i, :] - d2[i, :]) for i in range(35)]
    #     plt.title("Original")
    #     plt.plot(dists_o)
    #     plt.show()
    #     create_ani(d1, d2, "orig.gif")
    #
    #     # Filter out the 'problem' units to see if that fixes anything (large diff units, >=0.01)
    #     diff = d1 - d2
    #     probs = [np.abs(diff[i, :]) >= 0.01 for i in range(35)]
    #     iprobs = [np.where(p)[0] for p in probs]
    #     uprobs = []
    #     [uprobs.extend(l) for l in iprobs]
    #     uprobs2 = np.unique(uprobs)
    #     non_problems = np.array(list(set(list(range(327))).difference(set(list(uprobs2)))))
    #     d1p = d1[:, non_problems]
    #     d2p = d2[:, non_problems]
    #
    #     # Diff v time for all units, same plot
    #     diff_over_time(d1p, d2p, "Problem'd")
    #
    #     # To plot all original distances on same graph
    #     plt.clf()
    #     dists_p = [np.linalg.norm(d1p[i, :] - d2p[i, :]) for i in range(35)]
    #     plt.title("Filtered problems out")
    #     plt.plot(dists_p)
    #     plt.show()
    #     create_ani(d1p, d2p, "problemed.gif")
    #
    #     # Sort by highest summed total change in firing rate
    #     summed_diffs = np.sum([np.diff(np.abs(d1[:, i])) for i in range(d1.shape[1])], axis=1)
    #
    #     d1z = list(zip(summed_diffs, d1.swapaxes(0, 1)))
    #     d2z = list(zip(summed_diffs, d2.swapaxes(0, 1)))
    #
    #     d1sorted = sorted(d1z, key=lambda x: x[0])
    #     d2sorted = sorted(d2z, key=lambda x: x[0])
    #
    #     d1s = np.array([v[1] for v in d1sorted]).swapaxes(0, 1)
    #     d2s = np.array([v[1] for v in d2sorted]).swapaxes(0, 1)
    #     # To plot all distances on same graph
    #     plt.clf()
    #
    #     dists_s = [np.linalg.norm(d1s[i, :] - d2s[i, :]) for i in range(35)]
    #     plt.title("Sorted")
    #     plt.plot(dists_s)
    #     plt.show()
    #     create_ani(d1s, d2s, "sorted.gif")
    #     diff_over_time(d1s, d2s, "Sorted")
    #
    #     tw = 2

    def calculate(self, class_1_data, class_2_data):
        assert len(class_1_data.shape) == 2
        assert len(class_2_data.shape) == 2

        mean_1 = np.average(class_1_data, axis=1)  # expects shape to be (units, trials) pass in each time sep
        mean_2 = np.average(class_2_data, axis=1)  # Averaging over trials

        dist = np.linalg.norm(mean_1-mean_2)
        return dist
