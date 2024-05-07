import pendulum

# Recording consts
TOTAL_TRIAL_MS = 700
PRE_TRIAL_MS = 200
POST_TRIAL_MS = TOTAL_TRIAL_MS - PRE_TRIAL_MS
SPIKE_BIN_MS = 20

# Baseline consts
NUM_FIRINGRATE_SAMPLES = int(TOTAL_TRIAL_MS / SPIKE_BIN_MS)  # Should be 35
NUM_BASELINE_POINTS = 8  # First 8 points in a waveform will be used for z-scoring / baselining the waveform

# Unit filtering consts
TRIAL_THRESHOLD_SUM = 2.5  # NOT USED CURRENTLY Sum of all firing rates in all trials for a unit
UNIT_TRIAL_PERCENTAGE = .2  # NOT USED CURRENTLY Minimum percentage of trials that meet the threshold for a unit to include it

UNIT_ZETA_P_VALUE = 0.01  # p-value of a unit using the zeta test must be lower than this threshold

MOUSE_DETAILS = {
    "mlati7": {  # TODO Correct these values
        "birthday": pendulum.parse("5/2/22", strict=False),
        "strain": "mouse",
        "description": "this is a mouse",
        "sex": "M"
    }
}

SESSION_DESCRIPTION = "sess desc"  # TODO change me
EXPERIMENT_DESCRIPTION = "TODO"  # TODO Change me
EXPERIMENT_KEYWORDS = ["mouse", "neuropixels"]
EXPERIMENTERS = [
    "Hunt, Josh"
]

DEVICE_NAME = "neuropixels-probe"
DEVICE_DESCRIPTION = "neuropixels probe"
DEVICE_MANUFACTURER = "neuropixels"

METRIC_NAMES = {
    "ac": "amplitude_cutoff",
    "pr": "presence_ratio",
    "rpvr": "refactory_period_violation_rate",
    "fr": "global_firing_rate",
    "ql": "quality_labeling"
    # Quality labeling; 0 is multi-unit, 1 is single-unit; based on Kilosort and Anna's manual spike-sorting
}

METRIC_THRESHOLDS = {  # metric_name: func(unit_value) -> bool if true keep unit
    "amplitude_cutoff": lambda v: v <= 0.1,
    "presence_ratio": lambda v: v >= 0.9,
    "refactory_period_violation_rate": lambda v: v <= 0.5,  # isi
    "global_firing_rate": lambda v: v >= 0.2
    # don't include quality_labeling since that has a specific procedure
}
