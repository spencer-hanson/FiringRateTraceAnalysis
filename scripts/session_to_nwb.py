import os
import time

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
            data_files[file] = os.path.abspath(os.path.join(folder_path, file))
    return data_files


def main():
    # sessions_path = "google_drive/"  # Same folder lol
    sessions_path = "D:\\PopulationAnalysisRawHDF\\google_drive"  # NEEDS TO BE AN ABSOLUTE PATH
    data_files = check_for_data(sessions_path)
    force = False

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
    # data_files = {"05-16-2023-output.hdf": "05-16-2023-output.hdf"}
    # data_files = {"mlati6-2023-04-14-output.hdf": "google_drive/mlati6-2023-04-14-output.hdf"}
    # force = True
    # data_files = {"generated.hdf-nwb": "generated.hdf"}

    while True:
        print("Scanning for files to process..")
        for filename, filepath in data_files.items():
            try:
                print(f"Processing '{filename}'")
                name = ".".join(filename.split(".")[:-1])

                nwb_prefix = "D:\\PopulationAnalysisNWBs"
                name = os.path.join(nwb_prefix, name)

                if not os.path.exists(name):
                    os.mkdir(name)
                os.chdir(name)

                nwb_filename = f"{filename}-nwb.nwb"
                if os.path.exists(nwb_filename) and not force:
                    print("Already processed, skipping..")
                    os.chdir("../")
                    continue

                raw = HDFSessionProcessor(filepath, "mlati7", "session0")
                raw.save_to_nwb(nwb_filename, load_precalculated=True)
                del raw
                to_remove = ["calc_firingrates.npy", "calc_norm_firingrates.npy", "calc_rpperi_firingrates.npy", "calc_rpperi_norm_firingrates.npy", "calc_spike_trials.npy", "kilosort_firingrates.npy", "kilosort_spikes.npy", "calc_large_norm_firingrates.npy"]
                for fn in to_remove:
                    print(f"Removing {fn}..")
                    try:
                        os.remove(fn)
                    except Exception as e:
                        print(f"Couldn't remove '{fn}' Error: '{str(e)}'")

                os.chdir("../")
            except Exception as e2:
                os.chdir("../")
                # raise e2
                print(f"Error with file {filename} Skipping, Exception {e2}")
                fppp = open(f"error-{filename}.txt", "w")
                fppp.write(str(e2))
                fppp.close()
                continue

        print("Sleeping for 5 minutes before re-scanning..")
        time.sleep(60*5)  # Sleep for 5 minutes


if __name__ == "__main__":
    main()

