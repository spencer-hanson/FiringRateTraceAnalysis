import h5py
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.processors.nwb import NWBSessionProcessor


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    sess = NWBSessionProcessor("../scripts", filename, "../graphs")


    filename = "../scripts/output.hdf"
    data = h5py.File(filename)
    # TODO currently ignoring interpolation, dont think need to fix for visual
    passing_unit_filter = sess.qm_unit_filter().append(
        sess.probe_zeta_unit_filter()
    )

    probe_timestamps = sess.nwb.processing["behavior"]["probes"].data[:]
    saccade_timestamps = sess.nwb.processing["behavior"]["saccades"].data[:]

    all_uniq_timestamps = np.unique(np.array(data["spikes"]["timestamps"]))

    print("Starting..")

    chunks = 200
    for chunk in range(1, chunks + 1):
        chunk_size = round(len(np.array(data["spikes"]["clusters"])) / chunks)

        start_idx = (chunk-1)*chunk_size
        stop_idx = chunk*chunk_size
        spike_clusters = np.array(data["spikes"]["clusters"])[start_idx:stop_idx]
        spike_timestamps = np.array(data["spikes"]["timestamps"])[start_idx:stop_idx]
        uniq_nums = np.unique(spike_clusters)
        uniq_times = np.unique(spike_timestamps)
        unit_nums_timestamps = []  # (n, t) arr of each unit's timestamped spike times

        print("Creating unique time mapping")
        time_to_idx_mapping = {}  # Map the time to an index for plotting
        for idx, t in enumerate(uniq_times):
            time_to_idx_mapping[t] = idx

        print("Processing unit numbers..")
        for unit_num in uniq_nums:
            if unit_num not in passing_unit_filter.idxs():
                continue

            print(f"Unit num {unit_num}..")
            idxs = np.where(spike_clusters == unit_num)
            spike_times = spike_timestamps[idxs]

            spike_idxs = []
            for spike_time in spike_times:
                spike_idxs.append(time_to_idx_mapping[spike_time])

            unit_nums_timestamps.append(spike_idxs)

        saccade_lines = []
        probe_lines = []

        for probe_in_chunk in probe_timestamps:
            if probe_in_chunk in time_to_idx_mapping:
                probe_lines.append(time_to_idx_mapping[probe_in_chunk])

        for saccade_in_chunk in saccade_timestamps:
            if saccade_in_chunk in time_to_idx_mapping:
                saccade_lines.append(time_to_idx_mapping[saccade_in_chunk])

        fig, ax = plt.subplots(dpi=2000)

        print("Rendering raster..")
        ax.eventplot(unit_nums_timestamps, colors="black", lineoffsets=1, linelengths=1)
        for sac_line in saccade_lines:
            ax.vlines(sac_line, 0, len(unit_nums_timestamps), colors="red", linestyles="dashed")
        for pro_line in probe_lines:
            ax.vlines(pro_line, 0, len(unit_nums_timestamps), colors="blue", linestyles="solid")

        fig.suptitle("all_raster")
        fig.savefig(f"all_raster_{chunk}.png")
        # fig.show()
        plt.close(fig)
        tw = 2
        # judge_filters(sess)


if __name__ == "__main__":
    main()

