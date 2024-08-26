import pickle

import numpy as np
import matplotlib.pyplot as plt

from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.group import SessionGroup


def get_rpp_recalc():
    with open(f"newcalc_rp_peri-mlati10-2023-07-25-output.hdf.pickle", "rb") as f:
        rpp_recalculated = pickle.load(f)
    return rpp_recalculated


def plot_unit(sess: NWBSession, unit_num):
    units = sess.units()
    unit = units[unit_num]

    latency_start = -.3
    latency_end = -.2

    lt = sess.mixed_rel_timestamps >= latency_start
    gt = sess.mixed_rel_timestamps <= latency_end
    andd = np.logical_and(lt, gt)
    idxs = np.where(andd)[0]
    trial_latencies = sess.mixed_rel_timestamps[idxs]

    rpe_trfilt = sess.trial_filter_rp_extra().append(sess.trial_motion_filter(1))
    rpp_trfilt = sess.trial_filter_rp_peri(latency_start, latency_end, sess.trial_motion_filter(1))
    rmixed_idxs = sess.trial_filter_rmixed(latency_start, latency_end).idxs()[rpp_trfilt.idxs()]

    unit = unit[0]
    rmixed = unit[rmixed_idxs]
    rpextra = unit[rpe_trfilt.idxs()]
    rpp = sess.rp_peri_units()[unit_num][0][rpp_trfilt.idxs()]
    rpp_recalc = get_rpp_recalc()[unit_num][0]

    plt.plot(np.mean(rmixed, axis=0), color="green", label="Rmixed")
    plt.plot(np.mean(rpextra, axis=0), color="orange", label="RpExtra")
    plt.plot(np.mean(rpp, axis=0), color="blue", label="RpPeri")
    plt.plot(rpp_recalc, color="purple", label="RpPeriRC")

    # plt.title(f"Unit {unit_num}")
    plt.legend()
    plt.show()
    tw = 2


def debug_compare(sess: NWBSession):
    labels = sess.nwb.processing["behavior"]["unit_labels"].data[:]
    unit_num = np.where(labels == 406)[0]
    plot_unit(sess, unit_num)
    tw = 2


def main():
    grp = SessionGroup("D:\\PopulationAnalysisNWBs\\mlati10-2023-07-25-output*")
    filename, sess = next(grp.session_iter())
    debug_compare(sess)
    tw = 2


if __name__ == "__main__":
    main()
