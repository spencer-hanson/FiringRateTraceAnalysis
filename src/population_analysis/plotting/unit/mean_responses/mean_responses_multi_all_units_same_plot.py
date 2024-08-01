import glob
import os

import matplotlib.pyplot as plt

from population_analysis.plotting.unit.mean_responses.mean_responses_r_mixed import plot_r_mixed_mean_responses
from population_analysis.plotting.unit.mean_responses.mean_responses_rp_extra import plot_rp_extra_mean_responses
from population_analysis.plotting.unit.mean_responses.mean_responses_rp_peri import plot_rp_peri_mean_responses
from population_analysis.plotting.unit.mean_responses.mean_responses_rs import plot_rs_mean_responses
from population_analysis.processors.filters import BasicFilter
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_multi_mean_responses(sess, unit_filter):
    fig, axs = plt.subplots(2, 4, figsize=(32, 8))
    fig.subplots_adjust(wspace=0.6, hspace=.3)


    datas = [
        ("RpExtra", plot_rp_extra_mean_responses),
        ("Rs", plot_rs_mean_responses),
        ("Rmixed", plot_r_mixed_mean_responses),
        ("RpPeri", plot_rp_peri_mean_responses)
    ]

    idx = 0
    for name, func in datas:
        ax1 = axs[0, idx]
        ax2 = axs[1, idx]

        func(sess, unit_filter, [ax1, ax2])
        if idx == 0:
            ax1.set_ylabel("motion=-1")
            ax2.set_ylabel("motion=1")
        ax2.set_xlabel("Time (20ms bins)")

        ax1.set_title("")
        ax2.set_title("")
        ax1.set_title(f"{name} mean responses")
        idx = idx + 1
    plt.savefig("mean_responses.png")
    plt.show()
    tw = 2


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    nwbfiles = glob.glob("../../../../scripts/*/*.nwb")
    nwb_filename = nwbfiles[0]

    filepath = os.path.dirname(nwb_filename)
    filename = os.path.basename(nwb_filename)[:-len(".nwb")]

    sess = NWBSession(filepath, filename)

    unit_filter = BasicFilter.empty(sess.num_units)

    plot_multi_mean_responses(sess, unit_filter)


if __name__ == "__main__":
    main()
