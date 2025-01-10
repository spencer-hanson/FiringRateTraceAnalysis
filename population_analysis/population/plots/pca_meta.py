import warnings
from typing import Union, Any, Callable, Optional
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from sklearn.decomposition import PCA

"""
Meta information about PCA
"""


def run_pca(units: np.ndarray, components: Optional[int] = 3):
    # units is n x t (time by num units)
    pca = PCA(n_components=components)
    pca.fit(units)
    data = pca.transform(units)
    return pca, data


def pop_vec_pca_variances(units: np.ndarray, save_to_file: Union[bool, str] = False):
    # Variance of the PCA fit as the number of components are included, ie how much of the data is explained by the
    # the first N PCs

    pca, data = run_pca(units, None)
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
    # The first 3 PCs for a single unit, just to see it's 3d value as a bar graph

    pca, data = run_pca(units)

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
    # Graph of V x t (voltage by time) of the waveforms of the PCs that are the most important

    pca, data = run_pca(units)

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
    # Reconstruct the original waveform after reducing it with 'X' PCs, an animation as X goes from 1-35 (max)

    plt.clf()
    t_len = len(units[unit_idx])
    fig, ax = plt.subplots()

    def reconstruct(num_comps):

        pca, data = run_pca(units, components=num_comps)

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
