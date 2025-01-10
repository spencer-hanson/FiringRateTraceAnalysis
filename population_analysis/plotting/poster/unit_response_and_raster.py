import numpy as np

from population_analysis.plotting.unit.unit_summary import get_spike_idxs
from population_analysis.sessions.saccadic_modulation import NWBSession
import matplotlib.pyplot as plt


def plot_responses_and_rasters(sess: NWBSession, unit_num):
    fig, axs = plt.subplots(nrows=2)

    print("Creating trial filter..")
    # trial_filt = sess.trial_filter_rp_extra().append(sess.trial_motion_filter(1))
    trial_filt = sess.trial_filter_rmixed(-.2, .2, sess.trial_motion_filter(1))
    # name = "RpExtra"
    name = "RMixed"

    unitdata = sess.units()[:, trial_filt.idxs()][unit_num]
    response = np.mean(unitdata, axis=0)
    axs[1].plot([round(r, 2) for r in np.arange(-200, 500, 20)/1000], response)
    axs[1].set_ylim([np.min(response), 30])

    print("Getting and formatting spikes..")
    spikes = sess.spikes()
    spike_idxs = get_spike_idxs(spikes, unit_num, trial_filt.idxs())

    print("Plotting raster..")
    axs[0].eventplot(spike_idxs, colors="black", lineoffsets=1, linelengths=1)
    axs[0].set_title(f"{name} Response")
    axs[0].set_xlim([0, 700])
    axs[0].set_xticks([])

    axs[0].set_ylabel("Trial #")

    axs[1].set_xlabel("Time from probe (s)")

    plt.savefig("unit-raster.png", transparent=True)
    print("Saved!")


def main():
    nwbfile = "E:\\PopulationAnalysisNWBs\\mlati7-2023-05-15-output\\mlati7-2023-05-15-output.hdf.nwb"
    sess = NWBSession(nwbfile)
    unit_num = 344
    plot_responses_and_rasters(sess, unit_num)


if __name__ == "__main__":
    main()
