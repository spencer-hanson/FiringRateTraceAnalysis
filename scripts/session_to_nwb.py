import os
import warnings
import re

import h5py
import numpy as np

from population_analysis.consts import TOTAL_TRIAL_MS, PRE_TRIAL_MS, POST_TRIAL_MS
from population_analysis.population.units import UnitPopulation
from population_analysis.processors.raw import RawSessionProcessor

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
    data_files = {"idk": "output2.hdf"}

    for filename in list(data_files.values()):
        try:
            print(f"Processing '{filename}'")
            sess = RawSessionProcessor(filename, "mlati9")
            # +1 for leading \\, -4 for '.hdf'
            # nwb_filename = "_".join(re.split("\\\\|/", filename[len(SESSION_DATA_PATH)+1:]))[:-4] + ".nwb"
            # nwb_filename = "2023-05-15_mlati7_output_changed.nwb"  # TODO Remove me
            nwb_filename = "2023-06-30_mlati9_output.nwb"
            sess.save_to_nwb(nwb_filename, "session0")  # TODO change me
        except Exception as e:
            raise e
            # warnings.warn(f"Exception processing file '{filename}' skipping. Error: '{str(e)}'")


if __name__ == "__main__":
    main()

