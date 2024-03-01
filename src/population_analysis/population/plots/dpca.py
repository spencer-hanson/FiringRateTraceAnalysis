import warnings
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

"""
Graphs for dPCA (demixed PCA) 
See https://elifesciences.org/articles/10989
Library used https://github.com/machenslab/dPCA/tree/master/python
"""


def _get_dpca_data(num_components: int, grouped_units: np.ndarray):
    # Helper func to run dPCA on given units
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

    return dpca.fit_transform(grouped_units[0], grouped_units)
    # data["s"] - n_components x stims x time


def _do_dpca_plot(ax, graph_data, component_num, grouped_units: np.ndarray, stim_names: list[str], show=True):
    # Plot dPCA components, Time, Stimulus and Mixed, for a given component_num

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

    comp_loop_data = _get_dpca_data(default_n_components, np.array([grouped_units[0], grouped_units[0]]))
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

    siz_data = _get_dpca_data(1, np.array([grouped_units[0], grouped_units[0]]))
    siz_plotted = {"d": _do_dpca_plot(siz_axs, siz_data, 0, grouped_units, stim_names, show=False)}
    siz_fig.legend()

    def siz_loop(frame):
        [g.remove() for g in siz_plotted["d"]]
        siz_plotted["d"] = _do_dpca_plot(siz_axs, _get_dpca_data(frame % 35 or 1, np.array([grouped_units[0], grouped_units[0]])), 0, grouped_units, stim_names)
        return [*siz_axs, *siz_plotted["d"]]

    siz_ani = animation.FuncAnimation(fig=siz_fig, func=siz_loop, frames=35)
    siz_ani.save(filename=save_filename, writer="pillow")

