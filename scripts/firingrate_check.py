import h5py
import numpy as np

from population_analysis.processors.kilosort import KilosortProcessor


def main():
    raw_data = h5py.File("output.hdf")
    spike_clusters = np.array(raw_data["spikes"]["clusters"])
    spike_timings = np.array(raw_data["spikes"]["timestamps"])

    kp = KilosortProcessor(spike_clusters, spike_timings)

    fr, fr_bins = kp.calculate_firingrates(20, True)  # Bin size is 20ms in seconds
    sp, sp_bins = kp.calculate_spikes(True)

    tw = 2


if __name__ == "__main__":
    main()
