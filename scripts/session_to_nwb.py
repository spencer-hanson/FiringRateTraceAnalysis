import os

import h5py

from population_analysis.processors.experiments.saccadic_modulation.hdf import HDFSessionProcessor


def dictify_hd5(data):
    import h5py
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


def check_for_data(folder_path):
    data_files = {}

    for file in os.listdir(folder_path):
        if file.endswith(".hdf"):
            data_files[file] = os.path.join(folder_path, file)
    return data_files


def main():
    sessions_path = "../scripts"  # Same folder lol
    data_files = check_for_data(sessions_path)

    # data_files = {"idk": "E:\\PopulationAnalysis\\2023-05-15\\mlati7\\output.hdf"}
    # data_files = {
    #     # "idk": "../output-mlati6-2023-05-12.hdf"
    #     # "idk": "05-26-2023-output.hdf",
    #     # "idk": "05-15-2023-output.hdf",
    #     "generated.hdf": "../scripts\\generated.hdf",
    #     "generated2.hdf": "../scripts\\generated2.hdf"
    # }

    # dd = dictify_hd5(h5py.File("output.hdf"))
    # tw = 2

    for filename, filepath in data_files.items():
        try:
            print(f"Processing '{filename}'")
            name = ".".join(filename.split(".")[:-1])
            if not os.path.exists(name):
                os.mkdir(name)
            os.chdir(name)
            raw = HDFSessionProcessor("../" + filepath, "mlati7", "session0")
            raw.save_to_nwb(f"{filename}-nwb.nwb", load_precalculated=False)
            del raw
            os.chdir("../")
        except Exception as e:
            raise e
            # warnings.warn(f"Exception processing file '{filename}' skipping. Error: '{str(e)}'")


if __name__ == "__main__":
    main()

