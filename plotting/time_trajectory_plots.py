import os

from pynwb import NWBHDF5IO
import numpy as np

from population_analysis.population.plots.pca_meta import run_pca
from population_analysis.quantification import QuanDistribution
from population_analysis.quantification.euclidian import EuclidianQuantification
from matplotlib.patches import Ellipse
from matplotlib import animation
import matplotlib.pyplot as plt

from population_analysis.util import calc_num_bins

"""
Visualized PCA trajectories for different response types
"""

# def interpolate(pt1, pt2) -> np.ndarray:
#     x1, y1 = pt1[0], pt1[1]
#     x2, y2 = pt2[0], pt2[1]
#     dx = x2 - x1
#     dy = y2 - y1
#     num_samples = 25
#     dx = dx / num_samples
#     dy = dy / num_samples
#     points = []
#     for samp in range(num_samples):
#         points.append([x1+dx*samp, y1+dy*samp])
#     return np.array(points)


def quantify_timepoints(probe_units, saccade_units):
    from plotting.quantify_density_plots import graph_dists
    # for timepoint in range(35):
    timepoint = 0
    probe_timepoints = probe_units[:, :, timepoint]
    saccade_timepoints = saccade_units[:, :, timepoint]

    quan_name = "timepoints" + str(timepoint)
    quan = QuanDistribution(probe_timepoints, saccade_timepoints, EuclidianQuantification(quan_name))
    orig = quan.original()
    dists = quan.calculate()
    graph_dists(dists, orig, quan_name)
    tw = 2


def _all_pca_data_plot(axx, probe_pcad, saccade_pcad):
    # Plot the mean, the std circles, and all trial points all in one plot
    probe_mean = np.mean(probe_pcad, axis=0)
    saccade_mean = np.mean(saccade_pcad, axis=0)

    probe_std = np.std(probe_pcad, axis=0)
    saccade_std = np.std(saccade_pcad, axis=0)

    plots = [
        axx.scatter(probe_pcad[:, 0], probe_pcad[:, 1], color='bisque', alpha=0.8),
        axx.scatter(saccade_pcad[:, 0], saccade_pcad[:, 1], color='cornflowerblue', alpha=0.8),
        axx.scatter(*probe_mean, color='orange'), axx.scatter(*saccade_mean, color='blue')
    ]

    probe_ellipse = Ellipse(xy=probe_mean, width=probe_std[0], height=probe_std[1],
                            edgecolor='orange', fc='None', lw=2, label="Probe")
    saccade_ellipse = Ellipse(xy=saccade_mean, width=saccade_std[0], height=saccade_std[1],
                              edgecolor='blue', fc='None', lw=2, label="Saccade")

    plots.append(axx.add_patch(probe_ellipse))
    plots.append(axx.add_patch(saccade_ellipse))
    return plots


def ani_all_pca_plot(probe_units, saccade_units, pca, filename_prefix):
    print("Plotting all PCA graph..")
    pca_plot_data = {"data": []}
    fig = plt.figure()
    ax = fig.add_subplot()  # projection='3d')

    def pca_ani(frame):
        [v.remove() for v in pca_plot_data["data"]]
        pca_timepoint = frame % 35
        probe_pcad2 = pca.transform(probe_units[:, :, pca_timepoint].swapaxes(0, 1))
        saccade_pcad2 = pca.transform(saccade_units[:, :, pca_timepoint].swapaxes(0, 1))  # (trials, 2)
        plotted_pcas = _all_pca_data_plot(ax, probe_pcad2, saccade_pcad2)
        fig.suptitle(f"PCA of Saccade and Probe responses at timepoint {pca_timepoint}")
        pca_plot_data["data"] = plotted_pcas
        plt.legend()
        return pca_plot_data

    ani = animation.FuncAnimation(fig=fig, func=pca_ani, frames=35)
    ani.save(filename=f"{filename_prefix}/pca_timepoint_responses.gif", writer="pillow")


def ani_mean_trajectories(unit_dict, pca, filename_prefix, colormap, filename_suffix=""):
    print("Plotting mean trajectories..")
    num_timepoints = 35

    unit_avgs = {}
    for k, v in unit_dict.items():
        unit_avgs[k] = np.array([np.mean(pca.transform(v[:, :, tp].swapaxes(0, 1)), axis=0) for tp in range(num_timepoints)])

    pca_line_data = {"data": []}

    fig = plt.figure()
    ax = fig.add_subplot()

    def pca_line_ani(frame):
        [v.remove() for v in pca_line_data["data"]]
        pca_timepoint = frame % 35

        to_plot = []
        for pt in range(pca_timepoint + 1):
            if pt > 34 or pt == 0:
                continue

            for unit_name, unit_avg in unit_avgs.items():
                to_plot.extend(ax.plot(
                    [unit_avg[pt - 1][0], unit_avg[pt][0]],
                    [unit_avg[pt - 1][1], unit_avg[pt][1]],
                    color=colormap[unit_name],
                    linestyle="dashed"
                ))

        xmax = 0
        xmin = 0
        ymax = 0
        ymin = 0

        for unit_name, unit_avg in unit_avgs.items():
            xmax = max(unit_avg[:, 0].max(), xmax)
            xmin = min(unit_avg[:, 0].min(), xmin)
            ymax = max(unit_avg[:, 1].max(), ymax)
            ymin = min(unit_avg[:, 1].min(), ymin)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        for unit_name, unit_avg in unit_avgs.items():
            to_plot.append(ax.scatter(*unit_avg[pca_timepoint], color=colormap[unit_name], label=unit_name))

        fig.legend()
        pca_line_data["data"] = to_plot
        return to_plot

    fig.suptitle("Mean PCA response trajectory of Probe vs Saccade")
    ani = animation.FuncAnimation(fig=fig, func=pca_line_ani, frames=35)
    ani.save(filename=f"{filename_prefix}/timepoint_mean_trajectory{filename_suffix}.gif", writer="pillow")


def trial_trajectories_3d(pca_units, probe_units, saccade_units):
    print("Plotting each trial's trajectories in 3d")

    pca, data = run_pca(pca_units, components=3)
    num_timepoints = 35

    saccade_vals = np.array([pca.transform(saccade_units[:, :, tp].swapaxes(0, 1)) for tp in range(num_timepoints)])
    probe_vals = np.array([pca.transform(probe_units[:, :, tp].swapaxes(0, 1)) for tp in range(num_timepoints)])

    ax = plt.figure().add_subplot(projection='3d')
    colors = plt.get_cmap("hsv")
    for j in range(len(saccade_vals)):
        for i in range(1, num_timepoints):
            ax.plot(*saccade_vals[j][i - 1:i + 1].T, c="blue")

    for j in range(len(probe_vals)):
        for i in range(1, num_timepoints):
            ax.plot(*probe_vals[j][i - 1:i + 1].T, c="orange")

    plt.show()
    tw = 2


def mean_trajectories_3d(pca_units, probe_units, saccade_units):
    print("Plotting mean trajectories in 3d")

    pca, data = run_pca(pca_units, components=3)
    num_timepoints = 35

    saccade_avgs = np.array([np.mean(pca.transform(saccade_units[:, :, tp].swapaxes(0, 1)), axis=0) for tp in range(num_timepoints)])
    probe_avgs = np.array([np.mean(pca.transform(probe_units[:, :, tp].swapaxes(0, 1)), axis=0) for tp in range(num_timepoints)])

    arr_len = len(saccade_avgs)
    ax = plt.figure().add_subplot(projection='3d')
    for i in range(1, arr_len):
        ax.plot(*saccade_avgs[i - 1:i + 1].T, c="blue")

    for i in range(1, arr_len):
        ax.plot(*probe_avgs[i - 1:i + 1].T, c="orange")

    plt.show()
    tw = 2


def pca_variance_explained(pca_units):
    pca, _ = run_pca(pca_units, None)
    variance = pca.explained_variance_ratio_
    variance = np.cumsum(variance)
    fig, ax = plt.subplots()
    plt.title("PCA Explained Variance")
    plt.xlabel("Number of components")
    plt.ylabel("Percent of variance described")
    ax.plot(list(range(len(variance))), variance)
    plt.show()


def pca_components(pca):
    fig, ax = plt.subplots()
    colors = plt.get_cmap("Set1")
    for idx, component in enumerate(pca.components_):
        ax.plot(list(range(len(component))), component, color=colors(idx), label=f"{idx} PC")
    plt.show()


def unit_histograms(all_unit_data, timepoint, filename_prefix):
    # unit_data [(units,trials, t), ..]
    print("Plotting unit histogram animation")
    num_units = all_unit_data[0].shape[0]
    fig = plt.figure()
    ax = fig.add_subplot()
    to_remove = {"d": []}

    def update(frame):
        plotted = []
        [v.remove() for v in to_remove["d"]]
        if frame % 10 == 0:
            print(f"{frame}/{num_units}")

        unit_num = frame % num_units
        udata = [u[unit_num][:, timepoint] for u in all_unit_data]
        unit_data = np.hstack(udata)
        hist = np.histogram(unit_data, bins=calc_num_bins(unit_data), density=True)
        plotted.append(ax.stairs(hist[0], hist[1]))
        fig.suptitle(f"Unit {unit_num} hist at timepoint {timepoint}")
        to_remove["d"] = plotted
        return plotted

    ani = animation.FuncAnimation(fig=fig, func=update, frames=num_units)
    ani.save(filename=f"{filename_prefix}/unit_hist.gif", writer="pillow")
    tw = 2


def main():
    filename = "2023-05-15_mlati7_output"
    filepath = "../scripts/" + filename + ".nwb"
    filename_prefix = f"../graphs/{filename}"
    if not os.path.exists(filename_prefix):
        os.makedirs(filename_prefix)

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    probe_trial_idxs = nwb.processing["behavior"]["unit-trial-probe"].data[:]
    saccade_trial_idxs = nwb.processing["behavior"]["unit-trial-saccade"].data[:]
    mixed_trial_idxs = nwb.processing["behavior"]["unit-trial-mixed"].data[:]

    # Filter out mixed trials that saccades are more than 20ms away from the probe
    mixed_rel_timestamps = nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]
    mixed_filtered_idxs = np.abs(mixed_rel_timestamps) <= 0.02  # 20 ms
    mixed_trial_idxs = mixed_trial_idxs[mixed_filtered_idxs]

    # (units, trials, t)
    probe_units = nwb.units["trial_response_firing_rates"].data[:, probe_trial_idxs]
    saccade_units = nwb.units["trial_response_firing_rates"].data[:, saccade_trial_idxs]
    mixed_units = nwb.units["trial_response_firing_rates"].data[:, mixed_trial_idxs]
    rp_peri_units = nwb.units["r_p_peri_trials"].data[:]
    tw = 2

    num_units = probe_units.shape[0]
    to_pca_units = [probe_units, saccade_units, mixed_units, rp_peri_units]
    pca_units = [x.swapaxes(0, 2).reshape((-1, num_units)) for x in to_pca_units]
    pca_units = np.vstack(pca_units)

    pca, data = run_pca(pca_units, components=2)

    # pca_probe_units = probe_units.swapaxes(0, 2).reshape((-1, num_units))
    # pca_saccade_units = saccade_units.swapaxes(0, 2).reshape((-1, num_units))
    # pca_mixed_units = mixed_units.swapaxes(0, 2).reshape((-1, num_units))

    # Plots
    # ani_all_pca_plot(probe_units, saccade_units, pca, filename_prefix)
    colormap = {
        "Rp(Extra)": "red",
        "Rs": "blue",
        "Rmixed": "purple",
        "Rp(Peri)": "orange"
    }

    timepoint = 0
    unit_trial_data = [probe_units, saccade_units, mixed_units]
    # unit_histograms(unit_trial_data, timepoint, filename_prefix)  # TODO Sorta broken?

    ani_mean_trajectories(
        {"Rp(Extra)": probe_units, "Rs": saccade_units, "Rmixed": mixed_units, "Rp(Peri)": rp_peri_units}, pca,
        filename_prefix, colormap)

    ani_mean_trajectories(
        {"Rp(Extra)": probe_units, "Rp(Peri)": rp_peri_units}, pca,
        filename_prefix, colormap, filename_suffix="rpe_v_rpp")

    ani_mean_trajectories(
        {"Rmixed": mixed_units, "Rp(Peri)": rp_peri_units}, pca,
        filename_prefix, colormap, filename_suffix="rm_v_rpp")

    ani_mean_trajectories(
        {"Rs": saccade_units, "Rp(Peri)": rp_peri_units}, pca,
        filename_prefix, colormap, filename_suffix="rs_v_rpp")

    quantify_timepoints(probe_units, saccade_units)  # Prob dist
    mean_trajectories_3d(pca_units, probe_units, saccade_units)
    trial_trajectories_3d(pca_units, probe_units, saccade_units)
    pca_variance_explained(pca_units)
    pca_components(pca)


if __name__ == "__main__":
    main()
