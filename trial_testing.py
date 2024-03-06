import os
import h5py

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
        files = os.listdir(os.path.join(folder_path, mouse_folder))
        for file in files:
            if file.endswith(".hdf"):
                data_files[f"{os.path.basename(folder_path)}-{mouse_folder}"] = os.path.join(folder_path, mouse_folder, file)
    tw = 2


def main():
    data_files = {}
    for folder in os.listdir(SESSION_DATA_PATH):
        check_for_data(os.path.join(SESSION_DATA_PATH, folder), data_files)

    for k, v in data_files.items():
        data = dictify_hd5(h5py.File(v))
        tw = 2
    tw = 2
    pass


if __name__ == "__main__":
    main()

