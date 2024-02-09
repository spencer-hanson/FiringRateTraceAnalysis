from typing import Union, Any, Callable, Optional
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from sklearn.decomposition import PCA
from population_analysis.util import make_colormap


def pop_vec_traj(units: np.ndarray, colors: Optional[Callable[[int], Any]] = None, save_to_file: Union[bool, str] = False, plot_ax = None):
    # Units is an n x t arr, where n is a number of units, t is time

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


