import os

import h5py

from population_analysis.processors.experiments.saccadic_modulation.hdf import HDFSessionProcessor

SESSION_DATA_PATH = "E:\\PopulationAnalysis"


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


def check_for_data(folder_path, data_files):
    folders = os.listdir(folder_path)
    for mouse_folder in folders:
        if mouse_folder.startswith("mlati"):
            # TODO? Only use mlati mice
            files = os.listdir(os.path.join(folder_path, mouse_folder))
            for file in files:
                if file.endswith(".hdf"):
                    data_files[f"{os.path.basename(folder_path)}-{mouse_folder}"] = os.path.join(folder_path, mouse_folder, file)
    tw = 2


def main():
    data_files = {}
    # TODO Re-enable to process all sessions
    # for folder in os.listdir(SESSION_DATA_PATH):
    #     check_for_data(os.path.join(SESSION_DATA_PATH, folder), data_files)

    # data_files = {"idk": "E:\\PopulationAnalysis\\2023-05-15\\mlati7\\output.hdf"}
    data_files = {
        # "idk": "../output-mlati6-2023-05-12.hdf"
        "idk": "05-26-2023-output.hdf",
        # "idk": "generated.hdf"
    }

    # dd = dictify_hd5(h5py.File("output.hdf"))
    # tw = 2

    for filename in list(data_files.values()):
        try:
            print(f"Processing '{filename}'")
            name = ".".join(filename.split(".")[:-1])
            if not os.path.exists(name):
                os.mkdir(name)
            os.chdir(name)
            filename = f"../{filename}"
            raw = HDFSessionProcessor(filename, "mlati7", "session0")
            raw.save_to_nwb(f"{name}/{filename}-nwb.nwb", load_precalculated=False)
        except Exception as e:
            raise e
            # warnings.warn(f"Exception processing file '{filename}' skipping. Error: '{str(e)}'")


if __name__ == "__main__":
    main()

