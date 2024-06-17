import numpy as np
import pynwb
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from population_analysis.consts import PROBE_IDX
from population_analysis.processors.nwb import NWBSessionProcessor


class UnitNormalizer(object):
    def __init__(self, spike_clusters, spike_timestamps, trial_duration_event_idxs, unit_nums):
        self.clusts = spike_clusters
        self.unique_units = np.unique(self.clusts)
        self.timings = spike_timestamps
        self.full_events = trial_duration_event_idxs  # Expects [[start, event, stop], ..] for all trials
        self.unit_nums = unit_nums  # List of unit nums within clusts to loop over (since some are not present in second half of recording)
        self.num_trials = len(trial_duration_event_idxs)

    def _calc_firingrate(self, start_idx, stop_idx):
        units = []
        clusts = self.clusts[start_idx:stop_idx]
        timings = self.timings[start_idx:stop_idx]
        if len(clusts) == 0:
            return np.array([[0]]*len(self.unique_units))  # Return array of 0s for firing rate for all units

        bins = np.arange(timings[0], timings[-1], .2)  # TODO check we are binning correctly, make consts

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
        normalized_arr = np.empty((1,1))  # TODO want arr (trials, units, 35)

        unit_stds = {}  # unit_num: std TODO collect both motdirs for all units in all trials and average over trials
        all_trial_firingrate_as = []

        for trial_idx, event_data in enumerate(self.full_events):
            if trial_idx % int(self.num_trials/100) == 0:
                print(f"Normalizing trial {trial_idx}/{self.num_trials}..")

            trial_start_idx, trial_event_idx, trial_stop_idx = event_data
            timeperiod_a = (self.find_idx_from_relative_seconds(trial_event_idx, -20), self.find_idx_from_relative_seconds(trial_event_idx, -10))  # start, stop idx
            timeperiod_c = (trial_start_idx, trial_event_idx)

            firingrates_a = self._calc_firingrate(*timeperiod_a)  # TODO Fill in missing time periods with 0s
            firingrates_c = self._calc_firingrate(*timeperiod_c)
            event_firingrate = self.event_firingrates[trial_idx]  # TODO recalc firing rate

            all_trial_firingrate_as.append(firingrates_a)  # TODO split by motion direction type?

            for unit_idx in range(len(self.unit_nums)):  # TODO vectorize this loop
                mean = np.mean(firingrates_c[unit_idx])
                # Wait to divide by std until after we collect all data for all trials
                normalized_arr[trial_idx, unit_idx, :] = (event_firingrate[unit_idx] - mean)

                # gauss_a = self._gaussian_kernel_estimation(firingrates_a[unit_idx], 20)
                # gauss_c = self._gaussian_kernel_estimation(firingrates_c[unit_idx], .2)
                # gauss_event = self._gaussian_kernel_estimation(event_firingrate[unit_idx], .7)  # 700ms duration of event
                # amp = abs(event_firingrate[unit_idx, PROBE_IDX:] - mean).max()  # want the maximum amplitude after the probe
                # amp = amp if amp != 0 else 1
                # normalized_arr[idx, unit_idx, :] = (gauss_event - mean) / amp  # Divide by the max amplitude to normalize
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

        print("")
        return normalized_arr

