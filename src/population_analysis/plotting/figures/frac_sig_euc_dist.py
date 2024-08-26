import os

import numpy as np
from matplotlib import pyplot as plt

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import get_xaxis_vals
from population_analysis.plotting.distance.fraction_distance_significant import get_session_significant_timepoint_list
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.group import SessionGroup


def frac_sig_dist_euc(sess_group, confidence_val):
    os.chdir("../distance")
    quan = EuclidianQuantification()

    session_counts = np.zeros((NUM_FIRINGRATE_SAMPLES,))
    num_sessions = 0
    fig, ax = plt.subplots()

    for sess_namedata in sess_group.session_names_iter():

        counts = get_session_significant_timepoint_list(sess_namedata, quan, 1, confidence_val)
        if counts is not None:
            session_counts = session_counts + counts
            num_sessions = num_sessions + 1

    ax.title.set_text(f"Fraction of sessions outside of the {int(confidence_val*100)}th percentile baseline")
    # ax.title.set_text(f"{quan.get_name()} % sessions with distance above a {confidence_val} interval motion {motdir}")
    ax.plot(get_xaxis_vals(), session_counts/num_sessions)
    ax.set_ylabel("% of total sessions")
    ax.set_xlabel("Time (ms)")
    os.chdir("../figures")
    plt.savefig("euclidian-significance.svg")
    plt.show()

    tw = 2


def main():
    print("Loading group..")
    grp = SessionGroup("../../../../scripts")

    confidence_val = 0.99
    frac_sig_dist_euc(grp, confidence_val)


if __name__ == "__main__":
    main()
