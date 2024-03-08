import os
import h5py
import numpy as np

from population_analysis.consts import TOTAL_TRIAL_MS, PRE_TRIAL_MS, POST_TRIAL_MS
from population_analysis.population.units import UnitPopulation

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


def _extract_timestamp_idxs(spike_timestamps, other_timestamps) -> list[[float, float, float]]:
    # return indices into spike_timestamps within a window of -200ms to +700ms for trials times in other_timestamps
    idx_ranges = []

    other_len = len(other_timestamps)
    other_one_tenth = int(1/10 * other_len)
    for idx, ts in enumerate(other_timestamps):
        if idx % other_one_tenth == 0:
            print(f" {round(100*(idx / other_len), 3)}%", end="")
        if np.isnan(ts):
            # idx_ranges.append(None)
            continue
        start_idx = np.where(ts - (PRE_TRIAL_MS/1000) < spike_timestamps)[0][0]  # First index in tuple, first index is the edge
        end_idx = np.where(ts + (POST_TRIAL_MS/1000) <= spike_timestamps)[0][0]
        ts_idx = np.where(ts >= spike_timestamps)[0][-1]  # Index of the event timestamp itself, next smallest value
        idx_ranges.append([start_idx, ts_idx, end_idx])
    print("")
    return idx_ranges


def _remove_by_idxs(lst: list, idxs: list[int]) -> list:
    # Filter out a list by indexes to not include
    l2 = []
    for idx, l in enumerate(lst):
        if idx in idxs:
            continue
        else:
            l2.append(l)
    return l2


def _demix_trials(saccade_idxs: list[list[float]], probe_idxs: list[list[float]]):
    # If a saccade and probe trial occur within +- .5 sec (500ms) then they should be considered a mixed trial
    trials = {
        "saccade": [],
        "probe": [],
        "mixed": []
    }

    # Find probes that occur within 500ms of a saccade, remove them from the list of possible
    # saccades/probes after finding them
    def within_window(list1, list2, label1, label2):  # Find events within eachothers bounds
        mixed = []
        l1_to_remove = []
        l2_to_remove = []

        for f_idx, first_idx in enumerate(list1):
            f_start = first_idx[0]  # first start
            f_end = first_idx[2]

            for s_idx, second_idx in enumerate(list2):
                s_event = second_idx[1]  # second event time idx
                if f_start <= s_event <= f_end:  # Found mixed
                    mixed.append({
                        label1: first_idx,
                        label2: second_idx
                    })
                    l2_to_remove.append(s_idx)
                    l1_to_remove.append(f_idx)
                    break
            list2 = _remove_by_idxs(list2, l2_to_remove)
            l2_to_remove = []
        list1 = _remove_by_idxs(list1, l1_to_remove)
        return [list1, list2, mixed]

    # Find saccades that occur within 500ms of a probe
    print("Demixing saccades within 500ms from a probe")
    saccade_idxs, probe_idxs, both = within_window(saccade_idxs, probe_idxs, "saccade", "probe")

    # Find probes that occur within 500ms of a saccade
    print("Demixing probes within 500ms of a saccade")
    probe_idxs, saccade_idxs, both2 = within_window(probe_idxs, saccade_idxs, "probe", "saccade")

    trials["saccade"] = saccade_idxs
    trials["probe"] = probe_idxs
    trials["mixed"] = [*both, *both2]

    return trials


def _create_unit_population(spike_clusters: np.ndarray, spike_timestamps: np.ndarray, probe_timestamps: np.ndarray,
                            saccade_timestamps: np.ndarray):
    unit_pop = UnitPopulation(spike_timestamps, spike_clusters)

    print("Extracting saccade spike timestamps..")
    saccade_spike_range_idxs = _extract_timestamp_idxs(spike_timestamps, saccade_timestamps)
    print("Extracting probe spike timestamps..")
    probe_spike_range_idxs = _extract_timestamp_idxs(spike_timestamps, probe_timestamps)

    trials = _demix_trials(saccade_spike_range_idxs, probe_spike_range_idxs)

    unit_pop.add_probe_trials(trials["probe"])
    unit_pop.add_saccade_trials(trials["saccade"])
    unit_pop.add_mixed_trials(trials["mixed"])

    tw = 2

    print("Calculating firing rates for all trials and units")
    unit_pop.calc_firingrates()
    tw = 2
    # saccs = saccade_spike_range_idxs[0]
    # sac_range = spike_timestamps[saccs[0]:saccs[1]]
    # bin_range = np.arange(sac_range[0], sac_range[-1], .02)
    # TODO need to bin number of spikes rates by 20ms bins to get firing rate, for each probe and saccade instance
    # TODO make sure you're using spikes from particular units and not mixing them up
    tw = 2

    pass


def _extract_data(filename: str):
    data = h5py.File(filename)
    spike_clusters = np.array(data["spikes"]["clusters"])
    spike_timestamps = np.array(data["spikes"]["timestamps"])
    probe_timestamps = np.array(data["stimuli"]["dg"]["probe"]["timestamps"])
    saccade_timestamps = np.array(data["saccades"]["predicted"]["left"]["nasal"]["timestamps"])

    unit_pop = _create_unit_population(spike_clusters, spike_timestamps, probe_timestamps, saccade_timestamps)


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

