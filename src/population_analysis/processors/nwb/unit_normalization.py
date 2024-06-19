import numpy as np
import pynwb
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from population_analysis.consts import PROBE_IDX, NUM_FIRINGRATE_SAMPLES, SPIKE_BIN_MS
from population_analysis.processors.nwb import NWBSessionProcessor


class UnitNormalizer(object):
    def __init__(self, spike_clusters, spike_timestamps, trial_duration_event_idxs, unit_nums):
        self.clusts = spike_clusters
        self.unique_units = np.unique(self.clusts)
        self.timings = spike_timestamps
        self.full_events = trial_duration_event_idxs  # Expects [[start, event, stop], ..] for all trials
        self.unit_nums = unit_nums  # List of unit nums within clusts to loop over (since some are not present in second half of recording)
        self.num_trials = len(self.full_events)

    def _calc_firingrate(self, start_idx, stop_idx, num_bins=None):
        units = []
        clusts = self.clusts[start_idx:stop_idx]
        timings = self.timings[start_idx:stop_idx]
        if len(clusts) == 0:
            return np.array([[0]]*len(self.unique_units))  # Return array of 0s for firing rate for all units

        diff = timings[-1] - timings[0]
        if num_bins is None:
            bin_size = SPIKE_BIN_MS / 1000  # Use default bin size
            num_bins = int(diff/bin_size)
        else:  # Want a specific number of timepoints, need to slightly adjust binsize
            bin_size = diff/num_bins

        bins = np.arange(timings[0], timings[-1] + .001, bin_size)[:num_bins + 1]  # Plus one for the last edge

        for u in self.unit_nums:
            mask = np.where(clusts == u)[0]
            vals = timings[mask]
            hist, bin_edges = np.histogram(vals, bins=bins)
            rate = hist / 20  # 20ms bins
            units.append(rate)

        return np.array(units)

    # def _gaussian_kernel_estimation(self, arr, time_len_in_seconds):
    #     kernel_binsize = .02
    #
    #     if arr.min() == 0 and arr.max() == 0:
    #         return np.zeros((int(time_len_in_seconds/kernel_binsize)))
    #
    #     bandwith_sigma = 1
    #     # gauss = gaussian_kde(arr)
    #     # gauss.set_bandwidth(bandwith_sigma / arr.std())
    #     # times_to_sample = np.arange(0, time_len_in_seconds, kernel_binsize)
    #     # pred = gauss(times_to_sample)  # Smooth out the array using the gaussian while sampling from 0-time_len..
    #     # pred = ndimage.gaussian_filter1d(arr, bandwith_sigma)
    #     pred = gaussian_filter(arr, bandwith_sigma)
    #     # import matplotlib.pyplot as plt
    #     # fig, axs = plt.subplots(2, 1)
    #     # axs[0].plot(arr)
    #     # axs[1].plot(pred)
    #     # plt.show()
    #     return pred

    def find_idx_from_relative_seconds(self, start_idx, relative_seconds):
        # Find the index into the spike_timings that is relative seconds away from the start_idx
        # ie -10 sec from the start_idx time
        event_time = self.timings[start_idx]
        new_time = event_time + relative_seconds
        increment = int(relative_seconds / abs(relative_seconds))

        cur_val = event_time
        cur_idx = start_idx
        cur_diff = abs(new_time - event_time)

        while 0 < cur_val < self.timings[-1]:
            new_diff = abs(new_time - cur_val)

            if new_diff > cur_diff:  # If our estimate gets larger, then we've reached the max
                return cur_idx + (increment*-1)  # We missed it by one, go back one index

            cur_idx = cur_idx + increment
            cur_val = self.timings[cur_idx]

        ret = np.clip(cur_idx, 0, len(self.timings))
        return ret

    def normalize(self):
        # |--A-10sec---|--B-10sec---|-C-.2sec--|---Probe--|
        # baseline mean C
        # std over just A
        normalized_arr = np.empty((self.num_trials, len(self.unit_nums), NUM_FIRINGRATE_SAMPLES))

        unit_stds = {}  # unit_num: std TODO collect both motdirs for all units in all trials and average over trials
        all_trial_firingrate_as = []

        for trial_idx, event_data in enumerate(self.full_events):
        # for trial_idx, event_data in enumerate(self.full_events[:5]):  # TODO remove me testing only first 5 trials
            if trial_idx % int(self.num_trials/100) == 0:
                print(f"Normalizing trial {trial_idx}/{self.num_trials}..")

            trial_start_idx, trial_event_idx, trial_stop_idx = event_data
            timeperiod_a = (self.find_idx_from_relative_seconds(trial_event_idx, -20), self.find_idx_from_relative_seconds(trial_event_idx, -10))  # start, stop idx
            timeperiod_c = (trial_start_idx, trial_event_idx)

            firingrates_a = self._calc_firingrate(*timeperiod_a)  # TODO Fill in missing time periods with 0s
            firingrates_c = self._calc_firingrate(*timeperiod_c)
            event_firingrate = self._calc_firingrate(trial_start_idx, trial_stop_idx, 35)  # TODO make sure returns correct shape

            all_trial_firingrate_as.append(firingrates_a)  # TODO split by motion direction type?

            mean = np.mean(firingrates_c, axis=1)
            # Broadcast mean from (units,) to (units, 35) so we can subtract the mean from eachtimepoint of each unit, respectively, to vectorize this calculation
            # [:, None] adds a new axis, so (units,) -> (units, 1)
            mean = np.broadcast_to(mean[:, None], (len(self.unit_nums), NUM_FIRINGRATE_SAMPLES))

            normalized_arr[trial_idx, :, :] = event_firingrate - mean

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(3, 1)
            # axs[0].plot(gauss_event)
            # axs[1].plot(normalized)
            # axs[2].plot(event_firingrate[unit_idx])
            # plt.show()
            tw = 2

        # std = np.std(firingrates_a[unit_idx])
        # std = std if std != 0 else 1
        # TODO divide by std
        all_trial_firingrate_as = np.array(all_trial_firingrate_as)
        stds = np.std(np.mean(all_trial_firingrate_as, axis=2), axis=0)  # Standard deviations of the mean firinall units across all trials
        stds[stds == 0] = 1  # Replace 0 stds with 1 (very unlikely to happen just in case)
        stds = np.broadcast_to(stds[None, :, None], normalized_arr.shape)

        arr = normalized_arr / stds
        print("")
        return arr

