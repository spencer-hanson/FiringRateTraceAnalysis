import numpy as np
import pynwb
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from population_analysis.consts import PROBE_IDX
from population_analysis.processors.nwb import NWBSessionProcessor


class UnitNormalizer(object):
    def __init__(self, nwb: pynwb.NWBFile):
        self.nwb = nwb
        self.clusts = nwb.processing["behavior"]["spike_clusters"].data[:]
        self.unique_units = np.unique(self.clusts)
        self.timings = nwb.processing["behavior"]["spike_timestamps"].data[:]
        self.events = nwb.processing["behavior"]["trial_event_idxs"].data[:]
        self.event_firingrates = nwb.units["trial_response_firing_rates"].data[:]  # comes in as units, trials, t
        self.event_firingrates = np.swapaxes(self.event_firingrates, 0, 1)  # Easier to work with in trials, units, t

        self.approx_10sec = int(10/.001)  # Approximately 10 seconds in index values
        self.approx_20ms = int(.2/.001)  # Approx 20ms in index values

    def _calc_firingrate(self, start_idx, stop_idx):
        units = []
        clusts = self.clusts[start_idx:stop_idx]
        timings = self.timings[start_idx:stop_idx]
        if len(clusts) == 0:
            return np.array([[0]]*len(self.unique_units))  # Return array of 0s for firing rate for all units

        bins = np.arange(timings[0], timings[-1], .2)

        for u in self.unique_units:
            mask = np.where(clusts == u)[0]
            vals = timings[mask]
            hist, bin_edges = np.histogram(vals, bins=bins)
            rate = hist / 20  # 20ms bins
            units.append(rate)

        return np.array(units)

    def _gaussian_kernel_estimation(self, arr, time_len_in_seconds):
        kernel_binsize = .02

        if arr.min() == 0 and arr.max() == 0:
            return np.zeros((int(time_len_in_seconds/kernel_binsize)))

        bandwith_sigma = 1
        # gauss = gaussian_kde(arr)
        # gauss.set_bandwidth(bandwith_sigma / arr.std())
        # times_to_sample = np.arange(0, time_len_in_seconds, kernel_binsize)
        # pred = gauss(times_to_sample)  # Smooth out the array using the gaussian while sampling from 0-time_len..
        # pred = ndimage.gaussian_filter1d(arr, bandwith_sigma)
        pred = gaussian_filter(arr, bandwith_sigma)
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(arr)
        # axs[1].plot(pred)
        # plt.show()
        return pred

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
        normalized_arr = np.empty(self.event_firingrates.shape)
        standardized_arr = np.empty(self.event_firingrates.shape)

        filename = "2023-05-15_mlati7_output"
        sess = NWBSessionProcessor("../scripts", filename, "../graphs")
        tmp_filt = sess.unit_filter_qm().append(
        sess.unit_filter_probe_zeta().append(
                sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
            )
        )

        for idx, event in enumerate(self.events):
            if idx % int(len(self.events)/100) == 0:
                print(f"Normalizing event {idx}/{len(self.events)}..")

            timeperiod_a = (self.find_idx_from_relative_seconds(event, -20), self.find_idx_from_relative_seconds(event, -10))  # start, stop idx
            timeperiod_c = (self.find_idx_from_relative_seconds(event, -.2), event)

            firingrates_a = self._calc_firingrate(*timeperiod_a)
            firingrates_c = self._calc_firingrate(*timeperiod_c)
            event_firingrate = self.event_firingrates[idx]
            # for unit_idx in range(len(event_firingrate)):
            for unit_idx in tmp_filt.idxs():
                gauss_a = self._gaussian_kernel_estimation(firingrates_a[unit_idx], 20)
                gauss_c = self._gaussian_kernel_estimation(firingrates_c[unit_idx], .2)
                gauss_event = self._gaussian_kernel_estimation(event_firingrate[unit_idx], .7)  # 700ms duration of event

                mean = np.mean(gauss_c)
                std = np.std(gauss_a)
                std = std if std != 0 else 1

                amp = abs(event_firingrate[unit_idx, PROBE_IDX:] - mean).max()  # want the maximum amplitude after the probe
                amp = amp if amp != 0 else 1

                standardized_arr[idx, unit_idx, :] = (gauss_event - mean) / std
                normalized_arr[idx, unit_idx, :] = (gauss_event - mean) / amp  # Divide by the max amplitude to normalize

                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(3, 1)
                # axs[0].plot(gauss_event)
                # axs[1].plot(normalized)
                # axs[2].plot(event_firingrate[unit_idx])
                # plt.show()
                tw = 2
        print("")
        return normalized_arr, standardized_arr

