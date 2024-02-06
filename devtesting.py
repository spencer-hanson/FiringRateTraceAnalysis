import time

import h5py


def dictify_hd5(data):
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
    data = h5py.File("testdata/data1.hdf")
    dd = dictify_hd5(data)
    import matplotlib.pyplot as plt
    import numpy as np

    from firingrate_trace.plots import trace_correlogram, trace_changemaps, trace_3d_parametric, trace_shaped_path
    from firingrate_trace.plots import show_spline, trace_animated_shaped_path

    arr = data["rProbe"]["dg"]

    # for p in plots:
    #     p(arr["left"][0], arr["left"][1])

    units = np.array(dd["unitNumber"]).flatten()
    session_unit_counts = []
    session_idxs = []
    previous = 0
    for idx, u in enumerate(units):
        if u > previous:
            previous = u
        else:
            session_unit_counts.append(units[idx-1])
            session_idxs.append(max(idx-1, 0))
            previous = 0

    # plt.plot(list(range(len(units))), units)
    # plt.vlines(session_idxs, 0, 500, color="red")
    # plt.show()
    tw = 2

    """
    Use 'unitLabel' to find non NaN units that are more responsive

    """

    unit1_firingrate = arr["left"][7206]
    unit2_firingrate = arr["left"][26387]

    unit1_firingrate = np.array(unit1_firingrate)
    unit2_firingrate = np.array(unit2_firingrate)

    print("Plotting basic spline")
    show_spline(unit1_firingrate, save_to_file="spline.png")

    print("Plotting changemaps")
    trace_changemaps(unit1_firingrate, unit2_firingrate, save_to_file="changemap.png")
    trace_changemaps(unit1_firingrate, unit2_firingrate, spline=True, save_to_file="changemap_spline.png")

    print("Plotting shaped paths")
    trace_shaped_path(unit1_firingrate, unit2_firingrate, use_convex_hull=True, save_to_file="trace_hull.png")
    trace_shaped_path(unit1_firingrate, unit2_firingrate, use_convex_hull=False, fill=False, spline=False, save_to_file="trace_vanilla.png")
    trace_shaped_path(unit1_firingrate, unit2_firingrate, use_convex_hull=False, fill=False, spline=True, save_to_file="trace_spline.png")

    print("Plotting 3d parametrics")
    trace_3d_parametric(unit1_firingrate, unit2_firingrate, save_to_file="parametric.png")
    trace_3d_parametric(unit1_firingrate, unit2_firingrate, spline=True, save_to_file="parametric_spline.png")

    # start = 1
    # count = session_unit_counts[0]
    # # count = 10
    # trace_animated_shaped_path(
    #     arr["left"][0],
    #     arr["left"][1:count],
    #     start,
    #     list(range(2, count + 1))
    # )
    tw = 2
    pass


if __name__ == "__main__":
    main()

