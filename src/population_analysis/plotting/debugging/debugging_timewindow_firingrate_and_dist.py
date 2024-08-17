import os
import pickle

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import get_xaxis_vals, confidence_interval
from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import calc_quandist
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup
import matplotlib.pyplot as plt
import numpy as np


def rnd(x):
    return int(x*1000)


def sess_firingrate(rpp, rpe, ax):
    ax.plot(np.mean(np.mean(rpe, axis=1), axis=0), color="orange", label="RpExtra")
    ax.plot(np.mean(np.mean(rpp, axis=1), axis=0), color="blue", label="RpPeri")


def calc_rpextra_error_distribution(sess, use_cached, motdir):
    quan = EuclidianQuantification()
    tmp_fn = f"debugging_timewindow-{sess.filename_no_ext}.pickle"
    if os.path.exists(tmp_fn):
        with open(tmp_fn, "rb") as f:
            print("LOADING PRECALCULATED RPEXTRA DISTRIBUTION!")
            return pickle.load(f)

    ufilt = sess.unit_filter_premade()
    rpperi = sess.rp_peri_units().shape[1]
    rpextra = len(sess.trial_filter_rp_extra().idxs())
    prop = rpperi / rpextra
    prop = prop / 10  # divide by 10 since we have 10 latencies
    prop = prop / 2  # divide by 2 since we have 2 directions

    motions = [motdir]
    quan_dist_motdir_dict = calc_quandist(sess, ufilt, sess.trial_filter_rp_extra(), sess.filename_no_ext, quan=quan, use_cached=use_cached, base_prop=prop, motions=motions)
    data = quan_dist_motdir_dict[motions[0]]
    with open(tmp_fn, "wb") as f:
        pickle.dump(data, f)
    return data


def sess_distance(rpp, rpe, quan, motdir, confidence_val, ax, sess, use_cached):
    rpextra_error_distribution = calc_rpextra_error_distribution(sess, use_cached, motdir)

    distances = []
    means = []
    uppers = []
    lowers = []

    for t in range(NUM_FIRINGRATE_SAMPLES):
        lower, upper = confidence_interval(rpextra_error_distribution[:, t], confidence_val)
        mean = np.mean(rpextra_error_distribution[:, t], axis=0)
        distances.append(quan.calculate(rpp[:, :, t], rpe[:, :, t]))
        means.append(mean)
        uppers.append(upper)
        lowers.append(lower)

    ax.plot(get_xaxis_vals(), distances, color="blue")
    ax.plot(get_xaxis_vals(), means, color="orange")
    ax.plot(get_xaxis_vals(), uppers, color="orange", linestyle="dotted")
    ax.plot(get_xaxis_vals(), lowers, color="orange", linestyle="dotted")


def sess_summary(sess: NWBSession, filename, quan, motdir, confidence_val, use_cached):
    mmax = 10
    allfig, allax = plt.subplots(ncols=mmax, nrows=2, sharey="row", sharex="row", figsize=(16, 4))
    ufilt = sess.unit_filter_premade()
    rp_extra = sess.units()[ufilt.idxs()]
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]

    counts = {}
    for i in range(mmax):
        start = (i - (mmax / 2)) / 10
        end = ((i - (mmax / 2)) / 10) + .1

        rpe = rp_extra[:, sess.trial_filter_rp_extra().idxs()]
        rpp = rp_peri[:, sess.trial_filter_rp_peri(start, end).idxs()]
        title = f"{rnd(start)},{rnd(end)}"
        counts[title] = rpp.shape[1]

        allax[0][i].title.set_text(title)
        sess_firingrate(rpp, rpe, allax[0][i])
        sess_distance(rpp, rpe, quan, motdir, confidence_val, allax[1][i], sess, use_cached)

    allax[0][0].set_ylabel("Avg. Firing Rate")
    allax[0][1].set_ylabel("Distance")
    allfig.savefig(f"sess_debug/{filename}.png")
    plt.show()
    tw = 2


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("E:\\PopulationAnalysisNWBs")
    grp = NWBSessionGroup("C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\scripts\\nwbs\\mlati7-2023-05-15-output")
    if not os.path.exists("sess_debug"):
        os.mkdir("sess_debug")

    quan = EuclidianQuantification()
    motdir = 1
    confidence_val = 0.99
    use_cached = True

    for filename, sess in grp.session_iter():
        sess_summary(sess, filename, quan, motdir, confidence_val, use_cached)


if __name__ == "__main__":
    main()
