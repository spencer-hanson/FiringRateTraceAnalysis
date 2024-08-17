import math
import os.path
import warnings

import numpy as np


class KilosortProcessor(object):
    FIRING_RATE_PRECALCULATE_FILENAME = "kilosort_firingrates.npy"
    SPIKES_PRECALCULATE_FILENAME = "kilosort_spikes.npy"

    def __init__(self, spike_clusters, spike_timings):
        self.spike_clusters = spike_clusters
        self.unique_units = np.unique(self.spike_clusters)
        self.num_units = len(self.unique_units)
        self.spike_timings = spike_timings

    def _unit_firingrate(self, unit_num, bins, bin_size_ms):
        # Calculate the firingrate for unit 'unit_num' into the given bins
        unit_mask = np.where(self.spike_clusters == unit_num)[0]  # Indexes into spike_clusters and spike_timings where the specific unit number is
        unit_spike_times = self.spike_timings[unit_mask]
        hist, bin_edges = np.histogram(unit_spike_times, bins)
        hist = hist.astype("float64")
        hist /= bin_size_ms  # Divide by bin size in ms since sampling rate is 1ms
        return hist

    def calculate_firingrates(self, bin_size_ms, load_precalculated):
        # bin_size is in ms
        if bin_size_ms < 1:
            warnings.warn("Bin size cannot be < 1! Setting bin_size to 1ms")
            bin_size_ms = 1
        bin_size_seconds = bin_size_ms / 1000
        spike_start_time = np.min(self.spike_timings)
        spike_end_time = np.max(self.spike_timings)

        time_bins = np.arange(spike_start_time, spike_end_time, bin_size_seconds)
        if time_bins[-1] != spike_end_time:
            time_bins = np.append(time_bins, spike_end_time)  # Add end time if we don't cut exactly

        if load_precalculated:
            print("Attempting to load a precalculated firing rate from local directory..")
            if os.path.exists(KilosortProcessor.FIRING_RATE_PRECALCULATE_FILENAME):
                return np.load(KilosortProcessor.FIRING_RATE_PRECALCULATE_FILENAME, mmap_mode='r'), time_bins
            else:
                print(f"Precalculated file '{KilosortProcessor.FIRING_RATE_PRECALCULATE_FILENAME}' does not exist, generating..")

        print(f"Calculating firingrate of {len(self.unique_units)} Units and {len(self.spike_timings)} spikes, using a bin size of {bin_size_ms} ms")
        firing_rates = np.empty((self.num_units, len(time_bins) - 1))  # Subtract one to account for end bin edge
        for idx, unit_num in enumerate(self.unique_units):
            print(f"Processing Firingrate of Unit {idx}/{self.num_units}")
            firing_rates[idx, :] = self._unit_firingrate(unit_num, time_bins, bin_size_ms)

        print(f"Finished, writing to file '{KilosortProcessor.FIRING_RATE_PRECALCULATE_FILENAME}'..")
        np.save(KilosortProcessor.FIRING_RATE_PRECALCULATE_FILENAME, firing_rates)
        del firing_rates

        fr = np.load(KilosortProcessor.FIRING_RATE_PRECALCULATE_FILENAME, mmap_mode='r')
        return fr, time_bins

    def _round_float(self, flt):
        base = int(math.floor(flt))
        deciml = flt - base
        if deciml > .9:
            return round(flt)
        else:
            return int(flt)

    def calculate_spikes(self, load_precalculated):
        spike_start_time = np.min(self.spike_timings)
        spike_end_time = np.max(self.spike_timings)
        spike_bins = np.arange(spike_start_time, spike_end_time, 0.001)
        if spike_bins[-1] != spike_end_time:
            spike_bins = np.append(spike_bins, spike_end_time)  # Add end time if we don't cut exactly

        max_spikes = spike_bins.shape[0] - 1  # Number of ms in entire recording, minus one for the bin offset

        if load_precalculated:
            print("Attempting to load a precalculated spikes from local directory..")
            if os.path.exists(KilosortProcessor.SPIKES_PRECALCULATE_FILENAME):
                return np.load(KilosortProcessor.SPIKES_PRECALCULATE_FILENAME, mmap_mode='r')
            else:
                print(f"Precalculated file '{KilosortProcessor.SPIKES_PRECALCULATE_FILENAME}' does not exist, generating..")

        spikes = np.full((self.num_units, max_spikes), -1, dtype="int")  # Use -1 to double check after processing, that we didn't miss any spikes

        for idx, unit_num in enumerate(self.unique_units):
            print(f"Calculating spikes for unit {idx}/{self.num_units}")
            unit_spikes = self._unit_firingrate(unit_num, spike_bins, 1)
            unit_non_spikes = unit_spikes == 0
            unit_spike_idxs = np.where(np.logical_not(unit_non_spikes))[0]
            unit_nonspike_idxs = np.where(unit_non_spikes)[0]

            spikes[idx][unit_spike_idxs] = 1
            spikes[idx][unit_nonspike_idxs] = 0

            total = len(np.where(spikes[idx] == 1)[0]) + len(np.where(spikes[idx] == 0)[0])
            if total != max_spikes:
                raise ValueError("Error calculating spike timings! Missing data!")

        print(f"Finished, writing to file '{KilosortProcessor.SPIKES_PRECALCULATE_FILENAME}'..")
        np.save(KilosortProcessor.SPIKES_PRECALCULATE_FILENAME, spikes)
        del spikes
        sp = np.load(KilosortProcessor.SPIKES_PRECALCULATE_FILENAME, mmap_mode='r')

        return sp
