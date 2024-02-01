import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np


def trace_correlogram(unit1_firingrate: np.ndarray, unit2_firingrate: np.ndarray):
    plt.scatter(unit1_firingrate, unit2_firingrate)
    plt.show()


def trace_path(unit1_firingrate: np.ndarray, unit2_firingrate: np.ndarray):
    arr_len = len(unit1_firingrate)
    if len(unit1_firingrate.shape) != 1 or len(unit2_firingrate.shape) != 1:
        raise ValueError("Units must be one dimensional! unit1 shape {} unit2 shape {}".format(
            unit1_firingrate.shape,
            unit2_firingrate.shape
        ))

    if arr_len != len(unit2_firingrate):
        raise ValueError("Cannot plot the two arrays! Must be same len()! unit1 {} unit2 {}".format(
            arr_len,
            len(unit2_firingrate)
        ))

    # Path that looks like garbage lol
    # path_codes = [Path.MOVETO]
    # [path_codes.append(Path.LINETO) for _ in range(arr_len-1)]  # -2+1 for the start and finish, plus 1 for the duplicate ending point
    # unit1_firingrate = np.append(unit1_firingrate, unit1_firingrate[0])
    # unit2_firingrate = np.append(unit2_firingrate, unit2_firingrate[0])  # Add last point to close poly
    # # path_codes.append(Path.CLOSEPOLY)
    #
    # path = Path(list(zip(unit1_firingrate, unit2_firingrate)), path_codes)
    #
    # fig, ax = plt.subplots()
    # patch = patches.PathPatch(path, facecolor='orange', lw=2)
    # ax.add_patch(patch)
    # # ax.set_xlim(-2, 2)
    # # ax.set_ylim(-2, 2)
    # plt.show()

    # Parametric 3d plot
    # t = [i for i in range(arr_len)]
    # ax = plt.figure().add_subplot(projection='3d')
    # colors = plt.get_cmap("hsv")
    # for i in range(1, arr_len):
    #     ax.plot(unit1_firingrate[i - 1:i + 1], unit2_firingrate[i - 1:i + 1], t[i - 1:i + 1], c=colors(t[i-1]*4)[:3])
    # plt.show()

    # Plot the firing rate vs time for both, using a colormap of the other's value
    # fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
    # def plot_with_color(ax_idx, data, data2):
    #     t = np.array([i for i in range(arr_len)])
    #     norm = plt.Normalize(data2.min(), data2.max())
    #     points = np.array([t, data]).T.reshape(-1, 1, 2)
    #
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #     lc = LineCollection(segments, cmap='hsv', norm=norm)
    #     # Set the values used for colormapping
    #     lc.set_array(data2)
    #     lc.set_linewidth(2)
    #     line = axs[ax_idx].add_collection(lc)
    #     fig.colorbar(line, ax=axs[ax_idx])
    #     axs[ax_idx].set_xlim(t.min(), t.max())
    #     axs[ax_idx].set_ylim(data.min(), data.max())
    #
    # plot_with_color(0, unit1_firingrate, unit2_firingrate)
    # plot_with_color(1, unit2_firingrate, unit1_firingrate)
    # plt.show()
