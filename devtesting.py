

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


def main():
    import time
    import os
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    from population_analysis.util import filter_by_unitlabel
    from population_analysis.trajectory.plots import rate_correlogram, traj_changemaps, traj_3d_parametric
    from population_analysis.trajectory.plots import show_spline, traj_animated_shaped_path
    from population_analysis.trajectory.plots import traj_shaped_path
    from population_analysis.trajectory.population.plots import pop_vec_traj_clusters, pop_vec_traj, pop_vec_pcas
    from population_analysis.trajectory.population.plots import pop_vec_pca_variances, unit_pca_bar, unit_graph

    # filename = "2023-04-11.hdf"
    # filename = "2023-05-15.hdf"
    filename = "units_2024-02-08_f1.hdf"

    date_fn = filename.split(".")[0]

    data = h5py.File("testdata/{}".format(filename))

    # dd = dictify_hd5(data)
    # dg is drifing grating, fg..?
    # all_units = data["rProbe"]["dg"]["left"]

    baseline_num = 8

    all_units = data["rProbe"]["dg"]["left"]["fr"]  # Mean unit firing rates across trials
    units_len = len(all_units)
    # Unit means of first 8 timepoints to account for baseline
    all_units_mean = np.mean(all_units[:, :baseline_num], axis=1).reshape(units_len, 1)
    # Unit standard deviations of first 8 timepoints to account for baseline
    all_units_sd = np.mean(data["rProbe"]["dg"]["left"]["sd"][:, :baseline_num], axis=1).reshape(units_len, 1)
    all_units = (all_units - all_units_mean)/all_units_sd  # calc zscore

    tw = 2
    all_unit_labels = data["unitLabel"]
    # for p in plots:
    #     p(arr["left"][0], arr["left"][1])

    units = np.array(data["unitNumber"]).flatten()
    session_unit_counts = []
    session_idxs = []
    previous = -1
    for idx, u in enumerate(units):
        if u > previous:
            previous = u
        else:
            session_unit_counts.append(units[idx-1])
            session_idxs.append(max(idx-1, 0))
            previous = 0
    # Plot session unit numbers graph
    # plt.plot(list(range(len(units))), units)
    # plt.vlines(session_idxs, 0, 500, color="red")
    # plt.show()

    session_idxs.insert(0, 0)  # Insert 0 into the indexes since the 0th session has nothing
    session_num = 1

    session_units = all_units[session_idxs[session_num-1]: session_idxs[session_num]]
    session_units_cluster_labels = all_unit_labels[session_idxs[session_num-1]: session_idxs[session_num]]

    _unique_label_vals = np.unique(session_units_cluster_labels[:, 0])
    session_cluster_labels = _unique_label_vals[np.logical_not(np.isnan(_unique_label_vals))]
    clustered_units = []
    for label in session_cluster_labels:
        clustered_units.append(
            filter_by_unitlabel(session_units, session_units_cluster_labels, label)
        )

    save_prefix = f"graphs\\{date_fn}"
    if not os.path.exists(save_prefix):
        os.mkdir(save_prefix)

    # unit1_firingrate = all_units[7206]  # Units for 2023-04-11
    # unit2_firingrate = all_units[26387]

    unit1_firingrate = all_units[0]  # TODO pick out units to look at?
    unit2_firingrate = all_units[1]

    unit1_firingrate = np.array(unit1_firingrate)
    unit2_firingrate = np.array(unit2_firingrate)

    # print("Plotting cluster PCA trajectories gif")
    # pop_vec_traj_clusters(clustered_units, save_filename=f"{save_prefix}\\pop_vec_traj_clusters.gif")

    filtered_session_units = filter_by_unitlabel(session_units, session_units_cluster_labels, session_cluster_labels[1])

    pop_vec_pcas(filtered_session_units, save_to_file=f"{save_prefix}\\d_pcas.png")
    pop_vec_pca_variances(filtered_session_units, save_to_file=f"{save_prefix}\\pca_variances.png")
    unit_pca_bar(filtered_session_units, 0, save_to_file=f"{save_prefix}\\unit_pca_bar.png")
    unit_graph(filtered_session_units[0], save_to_file=f"{save_prefix}\\unit_graph.png")


    dasfadjgalskdjf
    # print("Plotting PCA trajectory for a session")
    # for i in range(len(session_cluster_labels)):
    #     session_units2 = filter_by_unitlabel(session_units, session_units_cluster_labels, session_cluster_labels[i])
    #     pop_vec_traj(session_units2.T, save_to_file=f"{save_prefix}\\pop_vec_traj_{i}.png")  # TODO transpose thingy
    # pop_vec_traj(session_units.T, save_to_file=False)

    print("Plotting basic spline")
    show_spline(unit1_firingrate, save_to_file=f"{save_prefix}\\spline.png")

    print("Plotting changemaps")
    traj_changemaps(unit1_firingrate, unit2_firingrate, save_to_file=f"{save_prefix}\\changemap.png")
    traj_changemaps(unit1_firingrate, unit2_firingrate, spline=True, save_to_file=f"{save_prefix}\\changemap_spline.png")

    print("Plotting shaped paths")
    traj_shaped_path(unit1_firingrate, unit2_firingrate, use_convex_hull=True, save_to_file=f"{save_prefix}\\trace_hull.png")
    traj_shaped_path(unit1_firingrate, unit2_firingrate, use_convex_hull=False, fill=False, spline=False, save_to_file=f"{save_prefix}\\trace_vanilla.png")
    traj_shaped_path(unit1_firingrate, unit2_firingrate, use_convex_hull=False, fill=False, spline=True, save_to_file=f"{save_prefix}\\trace_spline.png")

    print("Plotting 3d parametrics")
    traj_3d_parametric(unit1_firingrate, unit2_firingrate, save_to_file=f"{save_prefix}\\parametric.png")
    traj_3d_parametric(unit1_firingrate, unit2_firingrate, spline=True, save_to_file=f"{save_prefix}\\parametric_spline.png")

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
    pass


if __name__ == "__main__":
    main()

