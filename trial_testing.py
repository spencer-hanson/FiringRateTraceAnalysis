import os
import h5py
import numpy as np

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


def _extract_timestamp_idxs(spike_timestamps, other_timestamps):
    # return indices into spike_timestamps where other_timestamps are within a window of -200ms to +700ms
    idx_ranges = []
    print("Extracting timestamps..")
    other_len = len(other_timestamps)
    other_one_tenth = int(1/10 * other_len)
    for idx, ts in enumerate(other_timestamps):
        if idx % other_one_tenth == 0:
            print(f"{round(100*(idx / other_len), 3)}%")
        if np.isnan(ts):
            # idx_ranges.append(None)
            continue
        start_idx = np.where(ts - .2 < spike_timestamps)[0][0]  # First index in tuple, first index is the edge
        end_idx = np.where(ts + .7 <= spike_timestamps)[0][0]
        idx_ranges.append([start_idx, end_idx])
    return idx_ranges


def _extract_data(filename):
    data = h5py.File(filename)
    spike_clusters = np.array(data["spikes"]["clusters"])
    spike_timestamps = np.array(data["spikes"]["timestamps"])
    probe_timestamps = np.array(data["stimuli"]["dg"]["probe"]["timestamps"])
    saccade_timestamps = np.array(data["saccades"]["predicted"]["left"]["nasal"]["timestamps"])

    saccade_spike_range_idxs = _extract_timestamp_idxs(spike_timestamps, saccade_timestamps)
    probe_spike_range_idxs = _extract_timestamp_idxs(spike_timestamps, probe_timestamps)

    saccs = saccade_spike_range_idxs[0]
    sac_range = spike_timestamps[saccs[0]:saccs[1]]
    bin_range = np.arange(sac_range[0], sac_range[-1], .02)
    # TODO need to bin number of spikes rates by 20ms bins to get firing rate, for each probe and saccade instance
    # TODO make sure you're using spikes from particular units and not mixing them up
    tw = 2


def main():
    data_files = {}
    for folder in os.listdir(SESSION_DATA_PATH):
        check_for_data(os.path.join(SESSION_DATA_PATH, folder), data_files)

    _extract_data(list(data_files.items())[0][1])

    # for k, v in data_files.items():
    #     data = dictify_hd5(h5py.File(v))
    #     tw = 2
    tw = 2


if __name__ == "__main__":
    main()

