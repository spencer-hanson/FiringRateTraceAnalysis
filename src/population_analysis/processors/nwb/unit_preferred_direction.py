import numpy as np


class UnitPreferredDirection(object):
    def __init__(self, firing_rates, trial_motion_directions):
        self.firing_rates = firing_rates  # (units, trials, t)
        self.trial_mot_dir = np.array(trial_motion_directions)  # (trials,)

    def calculate(self):
        neg_dir_trials = self.firing_rates[:, self.trial_mot_dir == -1]
        pos_dir_trials = self.firing_rates[:, self.trial_mot_dir == 1]

        pos_unit_amps = np.max(np.mean(pos_dir_trials, axis=1), axis=1)
        neg_unit_amps = np.max(np.mean(neg_dir_trials, axis=1), axis=1)

        pos_idxs = np.where(pos_unit_amps > neg_unit_amps)[0]
        neg_idxs = np.where(np.logical_not(pos_unit_amps > neg_unit_amps))[0]
        dirs = np.zeros(pos_unit_amps.shape)

        dirs[pos_idxs] = 1
        dirs[neg_idxs] = -1
        num_neg, num_pos = np.sum(dirs + 1)/2, np.sum(dirs-1)/2
        print(f"Extracted unit preferred direction, num -1: {np.abs(num_neg)} num 1: {np.abs(num_pos)}")
        return dirs
