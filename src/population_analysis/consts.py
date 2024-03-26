import pendulum

# Recording consts
TOTAL_TRIAL_MS = 700
PRE_TRIAL_MS = 200
POST_TRIAL_MS = TOTAL_TRIAL_MS - PRE_TRIAL_MS
SPIKE_BIN_MS = 20

# Baseline consts
NUM_FIRINGRATE_SAMPLES = int(TOTAL_TRIAL_MS / SPIKE_BIN_MS)
NUM_BASELINE_POINTS = 8  # First 8 points in a waveform will be used for z-scoring / baselining the waveform

# Unit filtering consts
TRIAL_THRESHOLD_SUM = 0.01  # Sum of all firing rates in all trials for a unit
UNIT_TRIAL_PERCENTAGE = .2  # Minimum percentage of trials that meet the threshold for a unit to include it


MOUSE_DETAILS = {
    "mlati9": {  # TODO Correct these values
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