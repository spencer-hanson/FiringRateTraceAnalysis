import pickle
import matplotlib.pyplot as plt
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_null_dist(dist_filename: str):
    with open(dist_filename, "rb") as f:
        data = pickle.load(f)  # (10000, 35)

    time0 = data[:, 0]
    time12 = data[:, 12]

    fig, ax = plt.subplots()
    ax.hist(time0,  bins=200, color="salmon", label="-200ms from probe")
    ax.hist(time12, bins=200, color="orange", label="0ms from probe")
    ax.set_title("RpExtra Null Distribution")
    ax.set_xlabel("Euclidian distance")
    ax.legend()
    # plt.show()
    plt.savefig("null-distrib.png", transparent=True)

def main():
    fn = "E:\\PopulationAnalysisDists\\debugging_timewindow-mlati9-2023-07-05-output.hdf.pickle"
    plot_null_dist(fn)


if __name__ == "__main__":
    main()
