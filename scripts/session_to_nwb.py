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
    sessions_path = "D:\\PopulationAnalysisRawHDF"  # NEEDS TO BE AN ABSOLUTE PATH
    sessions_output_path = "D:\\PopulationAnalysisNWBs"

    data_files = check_for_data(sessions_path)
    force = False

    # dd = dictify_hd5(h5py.File("output.hdf"))
    # data_files = "mlati9-2023-07-14-output.hdf": "E:\\PopulationAnalysisRawHDF\\google_drive\\mlati9-2023-07-14-output.hdf"}
    data_files = {"mlati10-2023-07-25-output.hdf": "D:\\PopulationAnalysisRawHDF\\mlati10-2023-07-25-output.hdf"}
    # force = True

    # data_files = {"generated.hdf-nwb": "generated.hdf"}
    olddir = os.getcwd()

    while True:
        print("Scanning for files to process..")
        for filename, filepath in data_files.items():
            try:
                os.chdir(olddir)
                print(f"Processing '{filename}'")
                name = ".".join(filename.split(".")[:-1])
                name = os.path.join(sessions_output_path, name)
                if not os.path.exists(name):
                    os.mkdir(name)
                os.chdir(name)

                nwb_filename = f"{filename}.nwb"
                # if os.path.exists(nwb_filename) and not force:
                #     print("Already processed, skipping..")
                #     continue
                mouse_name = filename.split("-")[0]
                session_id = filename[len(mouse_name) + 1:-len("-output.hdf")]  # Chop off 'mlati8-' and '-output.hdf'

                raw = HDFSessionProcessor(filepath, mouse_name, session_id)
                raw.save_to_nwb(nwb_filename, load_precalculated=True)
                asdasdafasfasd
                del raw
                to_remove = ["calc_firingrates.npy", "calc_norm_firingrates.npy", "calc_rpperi_firingrates.npy", "calc_rpperi_norm_firingrates.npy", "calc_spike_trials.npy", "kilosort_firingrates.npy", "kilosort_spikes.npy", "calc_large_norm_firingrates.npy", "saccadic-trials.pickle"]
                for fn in to_remove:
                    print(f"Removing {fn}..")
                    try:
                        os.remove(fn)
                    except Exception as e:
                        print(f"Couldn't remove '{fn}' Error: '{str(e)}'")
            except Exception as e2:
                raise e2
                print(f"Error with file {filename} Skipping, Exception {e2}")
                fppp = open(f"error-{filename}.txt", "w")
                fppp.write(str(e2))
                fppp.close()
                continue

        print("Sleeping for 5 minutes before re-scanning..")
        raise ValueError("done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(60*5)  # Sleep for 5 minutes


if __name__ == "__main__":
    main()

