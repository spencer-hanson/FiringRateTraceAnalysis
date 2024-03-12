import pendulum

TOTAL_TRIAL_MS = 700
PRE_TRIAL_MS = 200
POST_TRIAL_MS = TOTAL_TRIAL_MS - PRE_TRIAL_MS
SPIKE_BIN_MS = 20
NUM_FIRINGRATE_SAMPLES = int(TOTAL_TRIAL_MS / SPIKE_BIN_MS)


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