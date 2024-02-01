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
    data = h5py.File("testdata/units_2024-01-24_f2.hdf")
    dd = dictify_hd5(data)
    from firingrate_trace.figures import trace_correlogram, trace_path
    arr = data["rProbe"]["dg"]
    trace_path(arr["left"][0], arr["left"][1])
    # trace_correlogram(arr["left"][0], arr["right"][0])
    tw = 2
    pass


if __name__ == "__main__":
    main()

