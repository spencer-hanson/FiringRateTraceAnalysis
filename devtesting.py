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

    # units = np.array(dd["unitNumber"]).flatten()
    # plt.plot(list(range(len(units))), units)
    # plt.show()
    tw = 2

    from firingrate_trace.plots import trace_correlogram, trace_heatmaps, trace_3d_parametric, trace_shaped_path, trace_multi_shaped_path
    plots = [
        trace_correlogram,
        trace_heatmaps,
        trace_3d_parametric,
        trace_shaped_path
    ]

    arr = data["rProbe"]["dg"]

    # for p in plots:
    #     p(arr["left"][0], arr["left"][1])

    # for i in range(25):
    #     print(i)
    #     trace_shaped_path(arr["left"][0], arr["left"][i+1])
    #
    trace_multi_shaped_path(arr["left"][0], arr["left"][1:25])

    # trace_path(arr["left"][0], arr["left"][1])
    # trace_correlogram(arr["left"][0], arr["right"][0])
    tw = 2
    pass


if __name__ == "__main__":
    main()

