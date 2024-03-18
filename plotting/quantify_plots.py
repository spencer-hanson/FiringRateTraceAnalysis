from pynwb import NWBHDF5IO

from population_analysis.quantification import QuanDistribution
from population_analysis.quantification.euclidian import EuclidianQuantification


def main():
    filepath = "../scripts/2023-05-15_mlati7_output.nwb"

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    probe_trial_idxs = nwb.processing["behavior"]["unit-trial-probe"].data[:]
    saccade_trial_idxs = nwb.processing["behavior"]["unit-trial-saccade"].data[:]

    probe_units = nwb.units["trial_firing_rates"].data[:, probe_trial_idxs]
    saccade_units = nwb.units["trial_firing_rates"].data[:, saccade_trial_idxs]

    # TODO Filter out 0's? Also do we concat trials? idk
    probe_units = probe_units.reshape((-1, 35))
    saccade_units = probe_units.reshape((-1, 35))

    epsilon = 0.00001  # TODO filter out 0s by filtering out the entries that sum up to lt or eq epsilon

    quan_dist = QuanDistribution(probe_units, saccade_units, EuclidianQuantification())
    dists = quan_dist.calculate()

    tw = 2


if __name__ == "__main__":
    main()

