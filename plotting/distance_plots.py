import os

import numpy as np
from pynwb import NWBHDF5IO

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.quantification.euclidian import EuclidianQuantification
import matplotlib.pyplot as plt


def _calc_dists(data_dict):
    # Calculate the mean euclidian distance between datasets, pairwise
    # data_dict is {"Rp(Extra)": <rp data>, ..}
    # returns like {"Rp(Extra)-Rp(Peri)": [dist_t_0, dist_t_1, ..], ..}
    euclid_dist = EuclidianQuantification("Pairwise")

    def dist_func(name, data, name1, data1):
        dist_data = []
        dist_name = f"{name}-{name1}"

        for t in range(NUM_FIRINGRATE_SAMPLES):
            dist_data.append(euclid_dist.calculate(
                data[:, :, t].swapaxes(0, 1),
                data1[:, :, t].swapaxes(0, 1)
            ))
        return dist_name, dist_data
    result = _pairwise_iter(data_dict, dist_func)
    d = {}
    for k, v in result:
        d[k] = v
    return d


def _pairwise_iter(data_dict, func):
    # func is f(name, data, name2, data2) -> result: Any
    results = []
    data_pairs = list(data_dict.items())
    for i in range(len(data_pairs)):
        name, data = data_pairs.pop()
        for name1, data1 in data_pairs:
            if name == name1:
                continue
            results.append(func(name, data, name1, data1))
    return results


def pairwise_mean_distances(data_dict):
    # Distance between the means of each data type (probe, saccade, mixed, etc..) over time
    dists = _calc_dists(data_dict)
    for pair_name, vals in dists.items():
        plt.plot(range(NUM_FIRINGRATE_SAMPLES), vals)
        plt.title(pair_name)
        plt.show()
    tw = 2


def pairwise_scaled_mean_distances_bootstrapped(data_dict):
    # iterate over a space of scales to determine distance between means

    scale_range = np.linspace(-2, 2)

    def iter_func(name1, data1, name2, data2):
        timepoint = 8
        data1_mean = np.mean(data1, axis=1)[:, timepoint]  # (units, t) avg trials out
        data2_mean = np.mean(data2, axis=1)[:, timepoint]

        data1_mean = scale_range * data1_mean[:, None]
        data2_mean = scale_range * data2_mean[:, None]

        euclid_dist = EuclidianQuantification("Scaled")
        dists = [euclid_dist.calculate(data1_mean[:, i], data2_mean[:, i]) for i in range(len(scale_range))]
        return f"{name1}-{name2}", dists

        # for t in range(NUM_FIRINGRATE_SAMPLES):
        #     t1 = data1_mean[:, t]
        #     t2 = data2_mean[:, t]

        # for unit_num in range(data1_mean.shape[0]):
        #     unit1 = data1_mean[unit_num]
        #     unit2 = data2_mean[unit_num]
        #     umin = min(np.min(unit1), np.min(unit2))
        #     umax = max(np.max(unit2), np.max(unit2))
        #     scale_range = np.linspace(umin, umax)

        # dmax =
        # dmin =
    pass
    result = _pairwise_iter(data_dict, iter_func)


    for k, v in result:
        ax = plt.figure().add_subplot(projection='3d')
        x = scale_range
        y = v
        # z =
        plt.plot(scale_range[range(len(v))], v)
        plt.title(k)
        plt.show()
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

    data_dict = {
            "Rp(Extra)": probe_units,  # (units, trials, t)
            "Rs": saccade_units,
            "Rmixed": mixed_units,
            "Rp(Peri)": rp_peri_units
        }

    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import axes3d

    ax = plt.figure().add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph
    # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

    ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
           xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()
    tw = 2
    # pairwise_mean_distances(data_dict)
    # pairwise_scaled_mean_distances_bootstrapped(data_dict)


if __name__ == "__main__":
    main()
