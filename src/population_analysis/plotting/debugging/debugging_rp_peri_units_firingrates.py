import pickle

import numpy as np

from population_analysis.processors.filters import BasicFilter
from population_analysis.processors.filters.trial_filters.rp_peri import RelativeTrialFilter
from population_analysis.sessions.saccadic_modulation import NWBSession
import matplotlib.pyplot as plt


def plot_unit_firingrates(units, axs, title):
    unit_means = np.mean(units, axis=1)
    for unit_idx in range(units.shape[0]):
        axs[0].plot(unit_means[unit_idx])

    axs[1].plot(np.mean(np.mean(units, axis=1), axis=0))
    axs[0].title.set_text(title)


def get_largest_unit_idxs(units):
    means = np.mean(units, axis=1)
    maxs = np.max(means, axis=1)
    vals = zip(range(maxs.shape[0]), maxs)
    srt = sorted(vals, key=lambda x: x[1])
    return np.array(srt[::-1][:])[:, 0].astype(int)


def get_rpp(sess: NWBSession, latency_start, latency_end, ufilt):
    # add_flt = None
    add_flt = sess.trial_motion_filter(1)
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]
    rpp_filt = sess.trial_filter_rp_peri(latency_start, latency_end, add_flt)
    rpp = rp_peri[:, rpp_filt.idxs()]
    return rpp


def get_rmixed(sess, latency_start, latency_end, ufilt):
    rmixed = sess.units()[ufilt.idxs(), :][:, sess.trial_filter_rmixed(latency_start, latency_end, sess.trial_motion_filter(1)).idxs()]
    return rmixed


def get_rs(sess: NWBSession, ufilt):
    trfilt = sess.trial_filter_rs().append(sess.trial_motion_filter(1))
    rs = sess.units()[ufilt.idxs()][:, trfilt.idxs()]
    return rs


def get_rpe(sess, ufilt):
    rpe_trfilt = sess.trial_filter_rp_extra().append(sess.trial_motion_filter(1))
    rpe = sess.units()[ufilt.idxs()][:, rpe_trfilt.idxs()]
    return rpe


def main():
    # fn = "mlati7-2023-05-12-output.hdf"
    fn = "mlati10-2023-07-25-output.hdf"
    # fpath = "D:\\PopulationAnalysisNWBs\\mlati7-2023-05-12-output"
    # fpath = "D:\\PopulationAnalysisNWBs\\mlati10-2023-07-25-output"
    fpath = "D:\\tmp"

    sess = NWBSession(fpath, fn)
    # ufilt = sess.unit_filter_premade()
    ufilt = BasicFilter.empty(sess.num_units)
    # ufilt = sess.unit_filter_probe_zeta()
    ufilt_idxs = ufilt.idxs()

    latency_start = -.3
    latency_end = -.2
    # latency_start = -.5
    # latency_end = .5
    # latency_start = .2
    # latency_end = .3

    rpp = get_rpp(sess, latency_start, latency_end, ufilt)
    rmixed = get_rmixed(sess, latency_start, latency_end, ufilt)
    rs = get_rs(sess, ufilt)
    rpe = get_rpe(sess, ufilt)

    # plot_unit_firingrates(rpp, axs)
    # plot_unit_firingrates(rmixed, axs)

    largest_unit_idxs = get_largest_unit_idxs(rs)
    labels = sess.nwb.processing["behavior"]["unit_labels"].data[:]
    largest_labels = labels[ufilt.idxs()[largest_unit_idxs]]
    unit_num = np.where(labels == 406)[0][0]
    largest_unit_idxs = [unit_num]

    rmixed = rmixed[largest_unit_idxs]
    rs = rs[largest_unit_idxs]
    rpp = rpp[largest_unit_idxs]
    rpe = rpe[largest_unit_idxs]

    # with open(f"newcalc_rp_peri-mlati7-2023-05-12-output.hdf.pickle", "rb") as f:
    with open(f"newcalc_rp_peri-{fn}.pickle", "rb") as f:
        rpp_recalculated = pickle.load(f)

    rpprc = rpp_recalculated[ufilt_idxs][largest_unit_idxs]
    # rpprc = rpprc[:, rpp_filt.idxs()]
    rpprc = rpprc[:, None, :]  # Add trials axis since we averaged it out in the recalculation
    units_to_plot = [
        (rmixed, "Rmixed"),
        (rs, "Rs"),
        (rpp, "RpPeri"),
        (rpprc, "RpPeriRC"),
        (rpe, "RpExtra")
    ]

    fig, axs = plt.subplots(nrows=2, ncols=len(units_to_plot), sharey="row", sharex=True)
    # fig, axs = plt.subplots(nrows=2, ncols=len(units_to_plot), sharey=False, sharex=True)
    fig.tight_layout()
    for i, unitdata in enumerate(units_to_plot):
        unitgroup, name = unitdata
        plot_unit_firingrates(unitgroup, [axs[0][i], axs[1][i]], name)

    # fig, axs = plt.subplots(nrows=4, ncols=3)
    # for motdir in [-1, None, 1]:
    #     if modir is None:
    #         mot_filt = None

    plt.show()
    tw = 2


if __name__ == "__main__":
    main()
