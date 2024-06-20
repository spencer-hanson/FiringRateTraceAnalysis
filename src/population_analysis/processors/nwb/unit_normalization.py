import numpy as np
import pynwb
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from population_analysis.consts import PROBE_IDX, NUM_FIRINGRATE_SAMPLES, SPIKE_BIN_MS
from population_analysis.processors.nwb import NWBSession


class UnitNormalizer(object):
    def __init__(self, spike_clusters, spike_timestamps, trial_duration_event_idxs, unit_nums, trial_motion_directions, preferred_unit_motions):
        self.clusts = spike_clusters
        self.unique_units = np.unique(self.clusts)
        self.timings = spike_timestamps
        self.full_events = trial_duration_event_idxs  # Expects [[start, event, stop], ..] for all trials
        self.unit_nums = unit_nums
        # List of unit nums within clusts to loop over (since some are not present in second half of recording)
        self.num_trials = len(self.full_events)
        self.trial_motion_dirs = np.array(trial_motion_directions)
        self.preferred_mot = preferred_unit_motions

    def _calc_firingrate(self, start_idx, stop_idx, num_bins):
        # (self.unit_nums, num_bins)
        units = []
        clusts = self.clusts[start_idx:stop_idx]
        timings = self.timings[start_idx:stop_idx]
        if len(clusts) == 0:
            raise ValueError("There are no spiking neurons during the requested period!")

        diff = timings[-1] - timings[0]
        bin_size = diff/num_bins

        bins = np.arange(timings[0], timings[-1] + .001, bin_size)[:num_bins + 1]  # Plus one for the last edge

        for u in self.unit_nums:
            mask = np.where(clusts == u)[0]
            vals = timings[mask]
            hist, bin_edges = np.histogram(vals, bins=bins)
            rate = hist / bin_size
            units.append(rate)

        return np.array(units)

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
        all_trial_firingrate_as = []

        for trial_idx, event_data in enumerate(self.full_events):
            if trial_idx % int(self.num_trials/100) == 0:
                print(f"Normalizing trial {trial_idx}/{self.num_trials}..")

            trial_start_idx, trial_event_idx, trial_stop_idx = event_data
            timeperiod_a = (self.find_idx_from_relative_seconds(trial_event_idx, -20), self.find_idx_from_relative_seconds(trial_event_idx, -10))  # start, stop idx
            timeperiod_c = (trial_start_idx, trial_event_idx)

            firingrates_a = self._calc_firingrate(*timeperiod_a, 1000)
            firingrates_c = self._calc_firingrate(*timeperiod_c, 10)
            event_firingrate = self._calc_firingrate(trial_start_idx, trial_stop_idx, 35)

            all_trial_firingrate_as.append(firingrates_a)

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

        all_trial_firingrate_as = np.array(all_trial_firingrate_as)

        for motdir in [-1, 1]:
            mot_std = all_trial_firingrate_as[self.trial_motion_dirs == motdir]
            pref_unit_idxs = np.where(self.preferred_mot == motdir)[0]

            # Standard deviations of the mean firinall units across all trials
            stds = np.std(np.mean(mot_std, axis=2), axis=0)[pref_unit_idxs]  # (units,)
            stds[stds == 0] = 1  # Replace 0 stds with 1 (very unlikely to happen just in case)
            stds = np.broadcast_to(stds[None, :, None], normalized_arr[:, pref_unit_idxs, :].shape)
            normalized_arr[:, pref_unit_idxs, :] = normalized_arr[:, pref_unit_idxs, :] / stds

        import matplotlib.pyplot as plt
        [plt.plot(g) for g in np.mean(normalized_arr[self.trial_motion_dirs == -1], axis=0)]

        print("")
        return normalized_arr

