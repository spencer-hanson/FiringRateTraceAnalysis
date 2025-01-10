import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.processors.filters import Filter
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_null_dist(dist_fn: str, dist_filename: str, timepoint, color, label):
    with open(dist_filename, "rb") as f:
        data = pickle.load(f)  # (10000, 35)
    with open(dist_fn, "rb") as f:
        distdata = pickle.load(f)

    dist = distdata[timepoint]
    time = data[:, timepoint]

    fig, ax = plt.subplots()
    ax.hist(time,  bins=200, color=color, label=label)
    ax.vlines(dist, 0, np.max(np.histogram(time, bins=200)[0]), color="black")
    ax.set_title("RpExtra Null Distribution")
    ax.set_xlabel("Euclidian distance")
    ax.legend()
    save_fn = f"null-distrib-{timepoint}.png"
    print(f"Saving {save_fn}..")
    plt.savefig(save_fn, transparent=True)


def main():
    prefix = "E:\\PopulationAnalysisDists"
    distrib_fn = os.path.join(prefix, "rpextra-quandistrib-2023-07-11.pickle")
    dist_fn = os.path.join(prefix, "euclidian-dist-2023-07-11-0.0.pickle")  # Latency 0 (100ms around probe)

    plot_null_dist(dist_fn, distrib_fn, 0, "salmon", "-200ms from probe")
    plot_null_dist(dist_fn, distrib_fn, 30, "orange", "100ms from probe")  # 30 is 20 for -200 + 100 for 10


if __name__ == "__main__":
    main()
