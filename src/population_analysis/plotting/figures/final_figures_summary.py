import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

prefix = "C:\\Users\\spenc\\Downloads"


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def plot_figure_F(filename):
    figdata = load_pickle(filename)
    rp_extra_path = figdata["rp_extra_path"]  # Array of (35, 3) - 35 timepoints, xyz
    rp_peri_path = figdata["rp_peri_path"]  # (35, 3)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(rp_peri_path[:, 0], rp_peri_path[:, 1], rp_peri_path[:, 2], label="RpPeri")
    ax.plot(rp_extra_path[:, 0], rp_extra_path[:, 1], rp_extra_path[:, 2], label="RpExtra")
    ax.title.set_text("FIGURE F - PCA NEURAL TRAJECTORY")
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()


def plot_figure_G(filename):
    figdata = load_pickle(filename)
    rpe_lower = figdata["rp_extra_lower_bound"]  # list of len 35
    rpe_mean = figdata["rp_extra_mean"]  # 35 len list
    rpe_upper = figdata["rp_extra_upper_bound"]  # 35 len list

    rpp_rpe_dist = figdata["rp_peri_v_rp_extra_euclidian_distance"]  # 35 len list

    fig, ax = plt.subplots()
    xvals = np.arange(-200, 500, 20)  # 35 bins of size 20ms with -200ms before and 500ms after probe
    ax.plot(xvals, rpe_lower, linestyle="dashed", color="orange")
    ax.plot(xvals, rpe_mean, color="orange", label="RpExtra")
    ax.plot(xvals, rpe_upper, linestyle="dashed", color="orange")

    ax.plot(xvals, rpp_rpe_dist, color="blue", label="Distance")

    ax.set_ylabel("Population euclidian distance")
    ax.set_xlabel("Time from probe (ms)")
    ax.title.set_text("FIGURE G - EUCLIDIAN DISTANCE")
    plt.show()
    plt.clf()
    plt.close()


def plot_figure_H(filename):
    figdata = load_pickle(filename)  # Arr (10,) for each latency
    fig, ax = plt.subplots()
    latencies = np.arange(-.5, .6, .1)
    rnd = lambda x: round(x, 2)
    # Create labels for latencies like '(-.5,-.4)', '(-.4,-.3)', ... '(.4,.5)'
    latency_labels = [f"({rnd(latencies[i])},{rnd(latencies[i+1])})" for i in range(figdata.shape[0])]
    ax.bar(latency_labels, figdata, width=0.5)
    ax.set_ylabel("Fraction of significant sessions")
    ax.set_xlabel("Saccade-Probe Latency (sec)")
    plt.xticks(rotation=90)
    ax.title.set_text("FIGURE H - FRACTION SIGNIFICANT P-VALUES p < 0.001")

    plt.show()
    plt.clf()
    plt.close()


def main():
    figure_f_filename = os.path.join(prefix, "FIG_F-neural-trajectories-data.pickle")
    figure_g_filename = os.path.join(prefix, "FIG_G-euclidian-distance-errorbars-data.pickle")
    figure_h_filename = os.path.join(prefix, "FIG_H-fraction-significant-pvalues-latency.pickle")

    plot_figure_F(figure_f_filename)
    plot_figure_G(figure_g_filename)
    plot_figure_H(figure_h_filename)
    print("Done!")


if __name__ == "__main__":
    main()
