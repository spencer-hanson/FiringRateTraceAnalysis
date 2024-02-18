import warnings
from typing import Union, Any, Callable, Optional
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from sklearn.decomposition import PCA
from population_analysis.util import make_colormap


def pop_vec_traj(units: np.ndarray, colors: Optional[Callable[[int], Any]] = None, save_to_file: Union[bool, str] = False, plot_ax = None):
    # Units is an t x n arr, where n is a number of units, t is time

    pca = PCA(n_components=3)
    pca.fit(units)
    data = pca.transform(units)

    if plot_ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        ax = plot_ax

    if colors is None:
        colors2 = plt.get_cmap("hsv")
        colors = make_colormap(lambda x: colors2(0), size=100)

    plots = []
    for i in range(1, len(data)):
        plots.append(
            *ax.plot(data[i - 1:i + 1, 0], data[i - 1:i + 1, 1], data[i - 1:i + 1, 2], c=colors((i * 3) % colors.N)[:3])
        )

    # Connect all points to all other points, almost like a polygon like thingy
    # for i in range(0, len(data)):
    #     for j in range(0, len(data)):
    #         plots.append(
    #             *ax.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], [data[i, 2], data[j, 2]], c=colors((i * 3) % colors.N)[:3])
    #         )

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        if plot_ax is None:
            plt.show()
    return plots


def pop_vec_traj_clusters(clustered_units: list[np.ndarray], save_filename: str):
    # clustered_units is a c x (n x t) arr, where c is clusters, n is num units, t is time
    colors = plt.get_cmap("hsv")
    clust_len = len(clustered_units)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    drawn = {"d": pop_vec_traj(clustered_units[0], colors=make_colormap(lambda x: colors(0), size=100), plot_ax=ax)}

    def update(frame):
        i = frame % clust_len
        [d.remove() for d in drawn["d"]]
        drawn["d"] = []
        drawn["d"].extend(
            pop_vec_traj(clustered_units[i], colors=make_colormap(lambda x: colors(i*10), size=100), plot_ax=ax)
        )

        return [ax, *drawn]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=clust_len)
    ani.save(filename=save_filename, writer="pillow")


def _pop_pca(units: np.ndarray, components: Optional[int] = 3):
    # units is n x t (time by num units)
    pca = PCA(n_components=components)
    pca.fit(units)
    data = pca.transform(units)
    return pca, data


def pop_vec_pca_variances(units: np.ndarray, save_to_file: Union[bool, str] = False):
    pca, data = _pop_pca(units, None)
    variance = pca.explained_variance_ratio_
    variance = np.cumsum(variance)
    fig, ax = plt.subplots()
    ax.plot(list(range(len(variance))), variance)

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()
    tw = 2


def unit_pca_bar(units: np.ndarray, unit_idx: int, save_to_file: Union[bool, str] = False):
    pca, data = _pop_pca(units)

    # Bar graph of the values of the PCA embeddings for unit 'unit_idx'
    fig, ax = plt.subplots()
    unit = data[unit_idx]

    x_vals = list(range(len(unit)))
    ax.bar(x_vals, unit, linewidth=0.4)
    ax.set_xticks(x_vals, [str(x) for x in x_vals])
    ax.set_xlabel("PC Number")
    ax.set_ylabel("PC Value")
    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()
    tw = 2


def pop_vec_pcas(units: np.ndarray, save_to_file: Union[bool, str] = False):
    # units is n x t (time by num units)
    pca, data = _pop_pca(units)

    fig, ax = plt.subplots()
    colors = plt.get_cmap("Set1")
    for idx, component in enumerate(pca.components_):
        ax.plot(list(range(len(component))), component, color=colors(idx), label=f"{idx} PC")

    fig.legend()
    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()
    tw = 2


def pop_vec_show_pca_reconstruction(units: np.ndarray, unit_idx: int, save_to_file: Union[bool, str] = False, animated: bool = False):
    plt.clf()
    t_len = len(units[unit_idx])
    fig, ax = plt.subplots()

    def reconstruct(num_comps):

        pca, data = _pop_pca(units, components=num_comps)

        pl = lambda d, n, c: ax.plot(list(range(len(d))), d, label=n, color=c)

        pca_single_unit = data[unit_idx]
        single_unit = units[unit_idx]

        components = pca.components_

        # rebuilt = pca.inverse_transform(pca_single_unit)
        rebuilt = np.sum(pca_single_unit.reshape(num_comps, 1) * components, axis=0)
        # rebuilt = np.dot(pca_single_unit, components)
        # rebuilt = rebuilt + pca.mean_
        single_unit = single_unit - pca.mean_  # subtract off mean to see comparison

        org = pl(single_unit, "original", "red")
        rec = pl(rebuilt, f"rebuilt with n={num_comps} PCs", "blue")
        return org, rec

    orig, recon = reconstruct(num_comps=3)

    if not animated:
        plt.legend()
        # [pl(c, f"comp{i}") for i, c in list(enumerate(components))[:3]]
        if save_to_file:
            plt.savefig(save_to_file)
        else:
            plt.show()
        tw = 2
    else:
        plt.title("PC reconstruction with 0 PCs")
        data = {"o": orig[0], "r": recon[0]}

        def update(frame):
            num = frame % t_len
            data["o"].remove()
            data["r"].remove()

            o, r = reconstruct(num)
            data["o"] = o[0]
            data["r"] = r[0]
            plt.title(f"PC reconstruction with {num} PCs")
            return [o, r]

        ani = animation.FuncAnimation(fig=fig, func=update, frames=t_len)
        ani.save(filename=save_to_file, writer="pillow")


def _get_dpca_data(num_components: int, grouped_units: np.ndarray):
    try:
        from dPCA.dPCA import dPCA  # Todo fix?
    except ImportError as e:
        warnings.warn(f"Unable to import dPCA package!")
        raise e

    # https://github.com/machenslab/dPCA
    dpca = dPCA(labels="st", regularizer="auto", n_components=num_components)
    dpca.protect = ["t"]
    # trial = samples x neurons x stimuli x time_points
    # all = neurons x stimuli x time_points
    # TODO reshape all to center mean? R -= mean(R.reshape((N,-1)),1)[:,None,None]

    return dpca.fit_transform(grouped_units[0], np.array([grouped_units[0], grouped_units[0]]))
    # data["s"] - n_components x stims x time


def _do_dpca_plot(ax, graph_data, component_num, grouped_units: np.ndarray, stim_names: list[str], show=True):
    time_range = np.arange(len(grouped_units[0][0][0]))  # Length of time
    num_stims = len(grouped_units[0][0])
    colors = plt.get_cmap("tab10")

    plots = []
    stim_names.extend([])
    for i, v in enumerate(["t", "s", "st"]):
        for s in range(num_stims):
            if i == 0:
                label = stim_names[i]
            else:
                label = None

            plots.append(ax[i].plot(time_range, graph_data[v][component_num, s], label=label, color=colors(s))[0])

        ax[i].set_title(f'#{component_num} {v} component')

    if show:
        plt.show()
    return plots


def pop_vec_dpca_fixed_comp_ani(grouped_units: np.ndarray, stim_names: list[str], save_filename: str):
    # dPCA fixed component animation through the components
    # grouped_units is a samples x neurons x stimuli x time_points array

    default_n_components = 12
    comp_loop_fig = plt.figure(figsize=(16, 7))
    comp_loop_fig.add_subplot(1, 3, 1)  # 1 row 3 cols 1st index to start
    comp_loop_fig.add_subplot(1, 3, 2)  # 1 row 3 cols 1st index to start
    comp_loop_fig.add_subplot(1, 3, 3)  # 1 row 3 cols 1st index to start

    comp_loop_axs = comp_loop_fig.get_axes()

    comp_loop_data = _get_dpca_data(default_n_components, grouped_units)
    comp_loop_plotted = {"d": _do_dpca_plot(comp_loop_axs, comp_loop_data, 0, grouped_units, stim_names, show=False)}
    comp_loop_fig.legend()

    def component_loop(frame):
        [g.remove() for g in comp_loop_plotted["d"]]
        comp_loop_plotted["d"] = _do_dpca_plot(comp_loop_axs, comp_loop_data, frame % default_n_components, grouped_units, stim_names)
        return [*comp_loop_axs, *comp_loop_plotted["d"]]

    comp_loop_ani = animation.FuncAnimation(fig=comp_loop_fig, func=component_loop, frames=default_n_components)
    comp_loop_ani.save(filename=save_filename, writer="pillow")

    # Save individual frames
    # for i in range(default_n_components):
    #     component_loop(i)
    #     comp_loop_fig.savefig(f"{save_prefix}/dpca_comps_{i}.png")


def pop_vec_dpca_component_iter_ani(grouped_units: np.ndarray, stim_names: list[str], save_filename: str):
    # dPCA components as n_components is iterated animation
    # grouped_units is a samples x neurons x stimuli x time_points array

    siz_fig = plt.figure(figsize=(16, 7))
    siz_fig.add_subplot(1, 3, 1)  # 1 row 3 cols - 1st index to start
    siz_fig.add_subplot(1, 3, 2)
    siz_fig.add_subplot(1, 3, 3)

    siz_axs = siz_fig.get_axes()

    siz_data = _get_dpca_data(1, grouped_units)
    siz_plotted = {"d": _do_dpca_plot(siz_axs, siz_data, 0, grouped_units, stim_names, show=False)}
    siz_fig.legend()

    def siz_loop(frame):
        [g.remove() for g in siz_plotted["d"]]
        siz_plotted["d"] = _do_dpca_plot(siz_axs, _get_dpca_data(frame % 35 or 1, grouped_units), 0, grouped_units, stim_names)
        return [*siz_axs, *siz_plotted["d"]]

    siz_ani = animation.FuncAnimation(fig=siz_fig, func=siz_loop, frames=35)
    siz_ani.save(filename=save_filename, writer="pillow")

