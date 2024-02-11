import math
from typing import Union, Any, Callable, Optional

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import splrep, BSpline
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from population_analysis.util import make_colormap


def rate_correlogram(unit1_firingrate: np.ndarray, unit2_firingrate: np.ndarray):
    plt.scatter(unit1_firingrate, unit2_firingrate)
    plt.show()


def _validate_traj_arrs(arr1: np.ndarray, arr2: np.ndarray):
    arr_len = len(arr1)

    if len(arr1.shape) != 1 or len(arr2.shape) != 1:
        raise ValueError("Units must be one dimensional! unit1 shape {} unit2 shape {}".format(
            arr1.shape,
            arr2.shape
        ))

    if arr_len != len(arr2):
        raise ValueError("Cannot plot the two arrays! Must be same len()! unit1 {} unit2 {}".format(
            arr_len,
            len(arr2)
        ))


def traj_changemaps(unit1_firingrate: np.ndarray, unit2_firingrate: np.ndarray, spline: bool = False, save_to_file: Union[bool, str] = False):
    _validate_traj_arrs(unit1_firingrate, unit2_firingrate)
    if spline:
        pointlist = _gen_unitpath(unit1_firingrate, unit2_firingrate, use_convex_hull=False, fill=False, spline=True)
        pointlist = pointlist.vertices
        unit1_firingrate = pointlist[:, 0]
        unit2_firingrate = pointlist[:, 1]
        tw = 2
    arr_len = len(unit1_firingrate)
    # Plot the firing rate vs time for both, using a colormap of the other's value
    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)

    def plot_with_color(ax_idx, data, data2):
        t = np.array([i for i in range(arr_len)])
        norm = plt.Normalize(data2.min(), data2.max())
        points = np.array([t, data]).T.reshape(-1, 1, 2)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='hsv', norm=norm)
        # Set the values used for colormapping
        lc.set_array(data2)
        lc.set_linewidth(2)
        line = axs[ax_idx].add_collection(lc)
        fig.colorbar(line, ax=axs[ax_idx])
        axs[ax_idx].set_xlim(t.min(), t.max())
        axs[ax_idx].set_ylim(data.min(), data.max())

    plot_with_color(0, unit1_firingrate, unit2_firingrate)
    plot_with_color(1, unit2_firingrate, unit1_firingrate)
    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()


def show_spline(unit1_firingrate: np.ndarray, save_to_file: Union[bool, str] = False):
    u1_len = len(unit1_firingrate)

    x_samps = np.linspace(0, u1_len-1, 1000)
    spline1 = splrep(list(range(u1_len)), unit1_firingrate, s=0)

    u1_samps = BSpline(*spline1)(x_samps)

    plt.plot(x_samps, u1_samps)
    plt.plot(list(range(u1_len)), unit1_firingrate)
    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()


def traj_3d_parametric(unit1_firingrate: np.ndarray, unit2_firingrate: np.ndarray, spline=False, save_to_file: Union[bool, str] = False):
    _validate_traj_arrs(unit1_firingrate, unit2_firingrate)

    if spline:
        pointlist = _gen_unitpath(unit1_firingrate, unit2_firingrate, use_convex_hull=False, fill=False, spline=True)
        pointlist = pointlist.vertices
        unit1_firingrate = pointlist[:, 0]
        unit2_firingrate = pointlist[:, 1]

    arr_len = len(unit1_firingrate)
    # Parametric 3d plot
    t = [i for i in range(arr_len)]
    ax = plt.figure().add_subplot(projection='3d')
    colors = plt.get_cmap("hsv")
    for i in range(1, arr_len):
        ax.plot(unit1_firingrate[i - 1:i + 1], unit2_firingrate[i - 1:i + 1], t[i - 1:i + 1], c=colors((t[i-1]*3) % colors.N)[:3])

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()


def _gen_unitpath(unit1_firingrate: np.ndarray, unit2_firingrate: np.ndarray, use_convex_hull=True, fill=True, spline=False):

    if spline:
        u1_len = len(unit1_firingrate)
        u2_len = len(unit2_firingrate)

        x_samps = np.linspace(0, u1_len - 1, 1000)
        spline1 = splrep(list(range(u1_len)), unit1_firingrate, s=0)
        spline2 = splrep(list(range(u2_len)), unit2_firingrate, s=0)

        unit1_firingrate = BSpline(*spline1)(x_samps)
        unit2_firingrate = BSpline(*spline2)(x_samps)

    pointlist = list(zip(unit1_firingrate, unit2_firingrate))
    pointlist = np.array(pointlist)
    if use_convex_hull:
        hull = ConvexHull(pointlist)
        pointlist = list(zip(pointlist[hull.vertices, 0], pointlist[hull.vertices, 1]))
        pointlist = np.array(pointlist)
    arr_len = len(pointlist)

    # Path that looks like garbage lol
    path_codes = [Path.MOVETO]
    [path_codes.append(Path.LINETO) for _ in range(arr_len-1)]  # Minus one to account for the MOVETO
    # Add last point to close polygon
    pointlist = np.append(pointlist, [np.array([unit1_firingrate[0], unit2_firingrate[0]])]).reshape(arr_len + 1, 2)
    pointlist = np.array(pointlist)
    if fill:
        path_codes.append(Path.CLOSEPOLY)
    else:
        path_codes.append(Path.LINETO)

    path = Path(pointlist, path_codes)

    return path


def traj_shaped_path(unit1_firingrate: np.ndarray, unit2_firingrate: np.ndarray, save_to_file: Union[bool, str] = False, use_convex_hull=True, fill=True, spline=False):
    _validate_traj_arrs(unit1_firingrate, unit2_firingrate)

    path = _gen_unitpath(unit1_firingrate, unit2_firingrate, use_convex_hull=use_convex_hull, fill=fill, spline=spline)

    _, ax = plt.subplots()
    patch = patches.PathPatch(path, facecolor='orange', lw=2)
    ax.add_patch(patch)

    u1 = np.array(path.vertices)[:, 0]
    u2 = np.array(path.vertices)[:, 1]

    bound = lambda x: (x.mean() - x.std()*3, x.mean() + x.std()*3)
    ax.set_xlim(bound(u1))
    ax.set_ylim(bound(u2))

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()


def _check_nan(unitdata):
    return any(np.isnan(unitdata))


def traj_animated_shaped_path(unit1_firingrate: np.ndarray, compare_units: np.ndarray, baseunit_num: int, other_unit_nums: list[int], filename: str):
    assert len(compare_units) == len(other_unit_nums), "Unit numbers don't match given number of compare_units!"

    # Normalize
    compare_units = compare_units / np.mean(compare_units, axis=-1).reshape(len(compare_units), 1)

    assert not _check_nan(unit1_firingrate), "Base unit contains a NaN value! Cannot compare!"
    comp = []

    for i in range(len(compare_units)):
        if _check_nan(compare_units[i]):
            print("Skipping unit '{}' contains a NaN value".format(i))
        else:
            comp.append(compare_units[i])
    compare_units = np.array(comp)

    colors = plt.get_cmap("hsv")
    colors_len = colors.N
    step = lambda x: int(math.ceil(colors_len/x))
    fig, ax = plt.subplots()

    polys = []
    lines = [4]

    for i in range(len(compare_units)):
        path = _gen_unitpath(unit1_firingrate, compare_units[i])
        patch = patches.PathPatch(path, facecolor=colors(i*step(len(compare_units))), lw=2)
        polys.append(patch)

        # linepath = _gen_unitpath(unit1_firingrate, compare_units[i], use_convex_hull=False, fill=False, spline=True)
        # lines.append(patches.PathPatch(linepath, facecolor="none", lw=1))

    pointlist = lambda x: np.array(list(zip(unit1_firingrate, compare_units[x])))

    current = {"poly": polys[0], "lines": lines[0]}
    ax.add_patch(current["poly"])
    # ax.add_patch(current["lines"])
    unitnum_fmt = f"Unit {baseunit_num} v {{}} Firing rates in 20ms bins"
    plt.title(unitnum_fmt.format(other_unit_nums[0]))

    scatter = ax.scatter(pointlist(0)[0], pointlist(0)[1], color="black", zorder=10)

    def update(func_frame):
        frame = func_frame % len(polys)
        plt.title(unitnum_fmt.format(other_unit_nums[frame]))

        current["poly"].remove()
        # current["lines"].remove()

        current["poly"] = polys[frame]
        # current["lines"] = lines[frame]

        ax.add_patch(current["poly"])
        # ax.add_patch(current["lines"])

        scatter.set_offsets(pointlist(frame))
        # return [current["poly"], scatter, current["lines"]]
        return [current["poly"], scatter]

    bounds = lambda x: [0, x.mean()+x.std()*4]
    # u1 = bounds(unit1_firingrate)
    # comp = bounds(compare_units)
    # limits = [min(u1[0], comp[0]), max(u1[1], comp[1])]
    # ax.set_xlim(*limits)
    # ax.set_ylim(*limits)
    ax.set_xlim(*bounds(unit1_firingrate))
    ax.set_ylim(*bounds(compare_units))
    ax.set_xlabel("Spikes / 20 ms (normalized)")
    ax.set_ylabel("Spikes / 20 ms (normalized)")
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(polys)*1)
    ani.save(filename=filename, writer="pillow")
    tw = 2

