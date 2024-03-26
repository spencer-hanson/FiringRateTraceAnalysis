from pynwb import NWBHDF5IO
import numpy as np

from population_analysis.quantification import QuanDistribution
from population_analysis.quantification.euclidian import EuclidianQuantification


def main():
    filepath = "../scripts/2023-05-15_mlati7_output.nwb"

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    probe_trial_idxs = nwb.processing["behavior"]["unit-trial-probe"].data[:]
    saccade_trial_idxs = nwb.processing["behavior"]["unit-trial-saccade"].data[:]

    probe_units = nwb.units["trial_response_firing_rates"].data[:, probe_trial_idxs]
    saccade_units = nwb.units["trial_response_firing_rates"].data[:, saccade_trial_idxs]

    timepoint = 0  # Start by looking at timepoint 0 in the response

    probe_timepoints = 0  # TODO get the points of the first timebin for all units, for each trial for probe and saccade
    saccade_timepoints = 0
    quan_name = "timepoints" + str(timepoint)

    quan = QuanDistribution(probe_timepoints, saccade_timepoints, EuclidianQuantification(quan_name))

    from plotting.quantify_density_plots import graph_dists
    orig = quan.original()
    dists = quan.calculate()

    graph_dists(dists, orig, quan_name)
    tw = 2


if __name__ == "__main__":
    main()
