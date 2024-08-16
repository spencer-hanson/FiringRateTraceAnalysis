import os
import pickle

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import get_xaxis_vals, confidence_interval
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup
import matplotlib.pyplot as plt
import numpy as np


def rnd(x):
    return int(x*1000)


def sess_firingrate(rpp, rpe, start, end, ax):
    ax.plot(np.mean(np.mean(rpe, axis=1), axis=0), color="orange", label="RpExtra")
    ax.plot(np.mean(np.mean(rpp, axis=1), axis=0), color="blue", label="RpPeri")
    ax.title.set_text(f"{rnd(start)},{rnd(end)}")


def sess_distance(start, end, quan, filename, motdir, confidence_val, ax):
    latency_key = f"{rnd(start)},{rnd(end)}"
    latency_dist_fn = f"{latency_key}-dists-{quan.get_name()}-{filename}-dir{motdir}.pickle"
    rpextra_error_distribution_fn = f"{filename}-{quan.get_name()}{motdir}.pickle"

    if os.path.exists(latency_dist_fn):
        print(f"Precalculated latency {latency_key} found..")
        with open(latency_dist_fn, "rb") as f:
            distances = pickle.load(f)

        with open(rpextra_error_distribution_fn, "rb") as f:
            rpextra_error_distribution = pickle.load(f)

        fig, oneax = plt.subplots()

        means = []
        uppers = []
        lowers = []
        for t in range(NUM_FIRINGRATE_SAMPLES):
            lower, upper = confidence_interval(rpextra_error_distribution[:, t], confidence_val)
            mean = np.mean(rpextra_error_distribution[:, t], axis=0)
            means.append(mean)
            uppers.append(upper)
            lowers.append(lower)

        ax.plot(get_xaxis_vals(), distances, color="blue")
        ax.plot(get_xaxis_vals(), means, color="orange")
        ax.plot(get_xaxis_vals(), uppers, color="orange", linestyle="dotted")
        ax.plot(get_xaxis_vals(), lowers, color="orange", linestyle="dotted")
        ax.title.set_text(latency_key)



def sess_summary(sess: NWBSession, filename, quan, motdir):
    mmax = 10
    allfig, allax = plt.subplots(ncols=mmax, nrows=2, sharey=True, sharex=True, figsize=(16, 4))
    ufilt = sess.unit_filter_premade()
    rp_extra = sess.units()[ufilt.idxs()]
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]


    for i in range(mmax):
        start = (i - (mmax / 2)) / 10
        end = ((i - (mmax / 2)) / 10) + .1

        rpe = rp_extra[:, sess.trial_filter_rp_extra().idxs()]
        rpp = rp_peri[:, sess.trial_filter_rp_peri(start, end).idxs()]
        sess_firingrate(rpp, rpe, start, end, ax)



    allfig.savefig(f"sess_debug/{filename}.png")
    # plt.show()
    # tw = 2


def main():
    print("Loading group..")
    grp = NWBSessionGroup("E:\\PopulationAnalysisNWBs")
    if not os.path.exists("sess_debug"):
        os.mkdir("sess_debug")

    quan = EuclidianQuantification()
    motdir = 1

    for filename, sess in grp.session_iter():
        sess_summary(sess, filename, quan, motdir)


if __name__ == "__main__":
    main()
