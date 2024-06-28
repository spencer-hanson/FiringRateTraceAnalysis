import numpy as np


class ModulationTrialGroup(object):
    def __init__(self, trial_list: list['ModulationTrial']):
        self._trials = np.array(trial_list)
        self.trial_labels = np.array([tr.trial_label for tr in trial_list])
        self.trial_motion_directions = np.array([tr.motion_direction for tr in trial_list])
        self.trial_types = np.unique(self.trial_labels)
        self.trial_type_idxs = {}

        for trial_type in self.trial_types:
            idxs = np.where(self.trial_labels == trial_type)[0]
            self.trial_type_idxs[trial_type] = idxs

        tw = 2

    def __str__(self):
        return f"TrialGroup({[(k,len(v)) for k,v in self.trial_type_idxs]})"

    def get_trials_by_type(self, trial_type):
        return self._trials[self.trial_type_idxs[trial_type]]

    def get_trial_idxs_by_motion(self, motion_direction):
        idxs = self.trial_motion_directions == motion_direction
        return idxs

    def get_trials_by_motion(self, motion_direction):
        idxs = self.get_trial_idxs_by_motion(motion_direction)
        return self._trials[idxs]

    @property
    def num_trials(self):
        return len(self._trials)

    def all_trials(self) -> np.ndarray:
        return self._trials


class ModulationTrial(object):
    def __init__(self, start_idx, end_idx, event_idx, event_time, trial_label, motion_direction, block_num, events):
        self.start_idx = start_idx  # indexes are into firing rate
        self.end_idx = end_idx
        self.event_idx = event_idx
        self.event_time = event_time  # Time in seconds of the event ie saccade/probe
        self.trial_label = trial_label  # 'saccade', 'probe' or 'mixed'
        self.motion_direction = motion_direction  # -1 or 1 (I don't know which is what)
        self.block_idx = block_num  # Which block this trial resides in (should be 0-60)
        self.events = events
        assert isinstance(self.events, dict)

    def __str__(self):
        return f"Trial({self.trial_label}, start={self.start_idx}, end={self.end_idx}, events_count={len(list(self.events.items()))}, dir={self.motion_direction})"

    def copy(self):
        tr = ModulationTrial(self.start_idx, self.end_idx, self.event_idx, self.event_time, self.trial_label, self.motion_direction, self.block_idx, self.events.copy())
        return tr

    def add_event(self, time_idx, label):
        self.events[label] = time_idx
