import numpy as np

from population_analysis.plotting.unit.unit_summary import get_spike_idxs
from population_analysis.sessions.saccadic_modulation import NWBSession
import matplotlib.pyplot as plt


def plot_responses(sess: NWBSession, unit_num):
    fig, ax = plt.subplots()

    print("Creating trial filters..")
    datas = [
        (sess.trial_filter_rp_extra().append(sess.trial_motion_filter(1)) , "orange", "RpExtra"),
        (sess.trial_filter_rp_peri(-.2, .2, sess.trial_motion_filter(1)), "blue", "RpPeri")
    ]

    for trial_filt, color, name in datas:
        unitdata = sess.units()[:, trial_filt.idxs()][unit_num]
        response = np.mean(unitdata, axis=0)

        ax.plot([round(r, 2) for r in np.arange(-200, 500, 20)/1000], response, color=color, label=name, linewidth=4)
    ax.set_yticks([])
    ax.set_ylabel("Normalized firing rate")
    ax.set_xlabel("Time from probe (s)")
    ax.set_title("RpExtra and RpPeri Responses")
    ax.legend()
    # plt.show()
    plt.savefig("responses.png", transparent=True)


def main():
    nwbfile = "E:\\PopulationAnalysisNWBs\\mlati7-2023-05-15-output\\mlati7-2023-05-15-output.hdf.nwb"
    sess = NWBSession(nwbfile)
    unit_num = 344
    plot_responses(sess, unit_num)


if __name__ == "__main__":
    main()
