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


def unit_graph(unit: np.ndarray, save_to_file: Union[bool, str] = False):
    # Graph of unit's firing rate
    fig, ax = plt.subplots()
    ax.plot(list(range(len(unit))), unit)
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


# def pop_vec_d_pcas(): TODO figure out use-case
#     try:
#         from dPCA.dPCA import dPCA  # Todo fix?
#     except ImportError as e:
#         warnings.warn(f"Unable to import dPCA package!")
#         raise e
#
#     dpca = dPCA(labels="st", n_components=3, regularizer="auto")
#     # trial = samples x neurons x stimuli x time_points
#     # all = neurons x stimuli x time_points
#     # TODO reshape all to center mean? R -= mean(R.reshape((N,-1)),1)[:,None,None]
#     # pca = dpca.fit_transform(all, trial)
#
#     tw = 2

