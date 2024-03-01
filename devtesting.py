import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from population_analysis.util import filter_by_unitlabel

from population_analysis.population.plots.dpca import pop_vec_dpca_fixed_comp_ani, pop_vec_dpca_component_iter_ani

from population_analysis.population.plots.pca_meta import pop_vec_pcas, pop_vec_pca_variances
from population_analysis.population.plots.pca_meta import unit_pca_bar, pop_vec_show_pca_reconstruction

from population_analysis.trajectory.plots import traj_changemaps, traj_3d_parametric
from population_analysis.trajectory.plots import show_spline, traj_animated_shaped_path
from population_analysis.trajectory.plots import traj_shaped_path



def dictify_hd5(data):
    import h5py
    if isinstance(data, h5py.Dataset):
        try:
            return list(data[:])
        except Exception as e:
            print(f"Errorrrrrr {str(e)}")
            return "BROKEN!!!!!!!!!!!!!!!!!!!!!!"
    else:
        dd = dict(data)
        d = {}
        for k, v in dd.items():
            d[k] = dictify_hd5(v)
        return d


def basic_plots(u: np.ndarray, save_prefix: str):
    print("Plotting basic spline")
    show_spline(u, save_to_file=f"{save_prefix}\\spline.png")


def changemap_plots(u1: np.ndarray, u2: np.ndarray, save_prefix: str):
    print("Plotting changemaps")
    traj_changemaps(u1, u2, save_to_file=f"{save_prefix}\\changemap.png")
    traj_changemaps(u1, u2, spline=True,
                    save_to_file=f"{save_prefix}\\changemap_spline.png")


def shaped_plots(u1: np.ndarray, u2: np.ndarray, save_prefix: str):
    print("Plotting shaped paths")
    traj_shaped_path(u1, u2, use_convex_hull=True, save_to_file=f"{save_prefix}\\trace_hull.png")
    traj_shaped_path(u1, u2, use_convex_hull=False, fill=False, spline=False, save_to_file=f"{save_prefix}\\trace_vanilla.png")
    traj_shaped_path(u1, u2, use_convex_hull=False, fill=False, spline=True, save_to_file=f"{save_prefix}\\trace_spline.png")


def parametric_plots(u1: np.ndarray, u2: np.ndarray, save_prefix: str):
    print("Plotting 3d parametrics")
    traj_3d_parametric(u1, u2, save_to_file=f"{save_prefix}\\parametric.png")
    traj_3d_parametric(u1, u2, spline=True, save_to_file=f"{save_prefix}\\parametric_spline.png")


def trajectory_animated_plots():  # TODO?
    t = traj_animated_shaped_path  # show as used import
    # start = 1
    # count = session_unit_counts[0]
    # # count = 10
    # trace_animated_shaped_path(
    #     all_units[0],
    #     all_units[1:count],
    #     start,
    #     list(range(2, count + 1))
    # )
    tw = 2


def pca_plots(units: np.ndarray, pca_example_unit_idx: int, save_prefix: str):
    print("Plotting PCA plots")
    pop_vec_pcas(units, save_to_file=f"{save_prefix}\\pca_pcs.png")
    pop_vec_pca_variances(units, save_to_file=f"{save_prefix}\\pca_variances.png")
    unit_pca_bar(units, pca_example_unit_idx, save_to_file=f"{save_prefix}\\pca_unit_bar.png")
    pop_vec_show_pca_reconstruction(units, pca_example_unit_idx, save_to_file=f"{save_prefix}\\pca_reconstruction.png")
    pop_vec_show_pca_reconstruction(units, pca_example_unit_idx, save_to_file=f"{save_prefix}\\pca_reconstruction_ani.gif", animated=True)
    tw = 2


def _hoist(data, keys: list[str]):
    d = data
    for k in keys:
        d = d[k]
    return d


def dpca_plots(data, save_prefix: str):
    # data["rProbe"]["dg"]["left"]["sd"]

    print("Plotting dPCA plots")

    stim_type = "dg"
    stim_list = [  # List of stimuli
        ["rProbe", stim_type, "left"],
        ["rProbe", stim_type, "right"],
        ["rSaccade", stim_type, "nasal"],
        ["rSaccade", stim_type, "temporal"]
    ]
    stim_datas = []  # stimuli x neurons x time_points
    # aaaa = _pull_data(data["rProbe"]["dg"]["left"], data["unitLabel"], data["unitNumber"])
    for stim in stim_list:
        stim_datas.append(
            _pull_data(_hoist(data, stim), data["unitLabel"], data["unitNumber"], session_num=1)
        )
    num_neurons = len(stim_datas[0])
    num_stim = len(stim_list)
    num_time = len(stim_datas[0][0])

    grouped_units = np.empty((1, num_neurons, num_stim, num_time))
    for stim_idx, stim_data in enumerate(stim_datas):
        grouped_units[0, :, stim_idx] = stim_data

    # need -> samples x neurons x stimuli x time_points array
    stim_names = ["probe_left", "probe_right", "saccade_nasal", "saccade_temporal"]

    pop_vec_dpca_component_iter_ani(grouped_units, stim_names, f"{save_prefix}\\dpca_comp_iter.gif")
    pop_vec_dpca_fixed_comp_ani(grouped_units, stim_names, f"{save_prefix}\\dpca_comp_fixed.gif")

    tw = 2


def plots(data, save_prefix):
    data_u = data["rProbe"]["dg"]["left"]
    units = _pull_data(data_u, data["unitLabel"], data["unitNumber"], session_num=1)

    pca_example_unit_idx = 0
    u1 = units[0]  # 7206  # Good units for 2023-04-11
    u2 = units[1]  # 26387

    # basic_plots(units[pca_example_unit_idx], save_prefix)
    # changemap_plots(u1, u2, save_prefix)
    # trajectory_animated_plots()  # TODO?
    # shaped_plots(u1, u2, save_prefix)
    # parametric_plots(u1, u2, save_prefix)
    # pca_plots(units, pca_example_unit_idx, save_prefix)
    dpca_plots(data, save_prefix)


def _get_session_idxs(units: np.ndarray) -> (list, list):
    session_unit_counts = []
    session_idxs = []
    previous = -1

    # Get the indicies for each session, based on the unit count
    for idx, u in enumerate(units):
        if u > previous:
            previous = u
        else:
            session_unit_counts.append(units[idx - 1][0])
            session_idxs.append(max(idx - 1, 0))
            previous = 0

    session_idxs.insert(0, 0)  # Insert 0 into the indexes since the 0th session has nothing
    session_unit_counts = np.array(session_unit_counts)
    return session_unit_counts, session_idxs


def _plot_unit_nums(unit_nums: np.ndarray, session_idxs: list[int]):
    # Plot session unit numbers graph
    plt.plot(list(range(len(unit_nums))), unit_nums)
    plt.vlines(session_idxs, 0, 500, color="red")
    plt.show()


def _get_unitdata(all_units_frs, all_units_sds, unit_labels, baseline_num_points: int) -> np.ndarray:
    all_units = all_units_frs  # All unit firing rates

    # Mean unit firing rates across trials
    units_len = len(all_units)

    # Unit means of first 8 timepoints to account for baseline
    all_units_mean = np.mean(all_units[:, :baseline_num_points], axis=1).reshape((-1, 1))

    # Unit standard deviations of first 8 timepoints to account for baseline
    all_units_sd = np.mean(all_units_sds[:, :baseline_num_points], axis=1).reshape((-1, 1))
    all_units = (all_units - all_units_mean) / all_units_sd  # calc zscore

    return all_units


def _load_data(filename: str):
    date_fn = filename.split(".")[0]
    save_prefix = f"graphs\\{date_fn}"
    if not os.path.exists(save_prefix):
        os.mkdir(save_prefix)

    return save_prefix, h5py.File("testdata/{}".format(filename))


def _get_clusters(units_data: np.ndarray, cluster_labels: np.ndarray) -> (np.ndarray, list[np.ndarray]):
    # Calculate labels for each cluster
    _unique_label_vals = np.unique(cluster_labels[:, 0])
    session_cluster_labels = _unique_label_vals[np.logical_not(np.isnan(_unique_label_vals))]
    clustered_units = []
    for label in session_cluster_labels:
        clustered_units.append(
            filter_by_unitlabel(units_data, cluster_labels, label)
        )

    return session_cluster_labels, clustered_units


def _pull_data(data, all_unit_labels, unit_nums, session_num):
    data_frs = data["fr"]
    data_sds = data["sd"]

    # Data
    baseline_num_points = 8  # Number of points to calculate the baseline mean for z-scoring the data
    all_units = _get_unitdata(data_frs, data_sds, all_unit_labels, baseline_num_points)

    # Unit Numbers and Sessions
    session_unit_counts, session_idxs = _get_session_idxs(unit_nums)
    # _plot_unit_nums(unit_nums, session_idxs)

    # Session units
    session_units = all_units[session_idxs[session_num-1]: session_idxs[session_num]]
    session_units_cluster_labels = all_unit_labels[session_idxs[session_num-1]: session_idxs[session_num]]
    session_cluster_labels, clustered_units = _get_clusters(session_units, session_units_cluster_labels)
    # NOTE clustered_units is [<n*t numpy arr>, ...] where n is the number of units in the cluster, and t is time
    # NOTE session_cluster_labels is a list of possible labels, eg [0, 1, 2, .., 9]

    # plot_units = filter_by_unitlabel(session_units, session_units_cluster_labels, session_cluster_labels[session_num])
    plot_units = np.vstack(clustered_units)  # TODO currently using all units
    return plot_units


def main():
    # filename = "2023-04-11.hdf"
    # filename = "2023-05-15.hdf"
    filename = "units_2024-02-08_f1.hdf"
    save_prefix, data = _load_data(filename)

    plots(data, save_prefix)


if __name__ == "__main__":
    main()

