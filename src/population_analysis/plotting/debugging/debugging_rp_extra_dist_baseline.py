import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import calc_quandist
from population_analysis.processors.filters import BasicFilter
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup

DISTANCES_LOCATION = "D:\\PopulationAnalysisDists"


def debug_rpe_baseline(folder, filename, use_cached):
    olddir = os.getcwd()
    os.chdir(DISTANCES_LOCATION)
    motdir = 1
    quan = EuclidianQuantification()

    if use_cached and os.path.exists(f"{filename}-{quan.get_name()}{motdir}.pickle"):
        print(f"Distance file already exists for '{filename}' skipping!")
        os.chdir(olddir)
        return

    try:
        print(f"Distance not found for session '{filename}', generating..")
        sess = NWBSession(folder, filename)

        ufilt = sess.unit_filter_premade()

        rpperi = sess.rp_peri_units().shape[1]
        rpextra = len(sess.trial_filter_rp_extra().idxs())
        prop = rpperi / rpextra
        prop = prop / 10  # divide by 10 since we have 10 latencies
        prop = prop / 2  # divide by 2 since we have 2 directions  TODO find proportion of directions
        rpp_intervals = {}
        rpp_names = {}
        rpp_latencies = {}

        mmax = 10
        for i in range(mmax):
            st = (i - (mmax / 2)) / 10
            end = ((i - (mmax / 2)) / 10) + .1
            rnd = lambda x: int(x * 1000)

            lt = sess.mixed_rel_timestamps >= st
            gt = sess.mixed_rel_timestamps <= end
            andd = np.logical_and(lt, gt)
            latency_key = f"{rnd(st)},{rnd(end)}"
            rpp_intervals[i] = andd
            rpp_names[i] = latency_key
            rpp_latencies[i] = (st, end)

        motions = [1]
        quan_dist_motdir_dict = calc_quandist(sess, ufilt, sess.trial_filter_rp_extra(), sess.filename_no_ext, quan=quan, use_cached=use_cached, base_prop=prop, motions=motions)
        # fig, ax = plt.subplots()
        # ax.plot(np.mean(quan_dist_motdir_dict[1], axis=0))
        # plt.show()
    except Exception as e:
        # raise e
        print(f"Error in session '{sess.filename_no_ext}' Error: '{str(e)}'")
        with open(f"{sess.filename_no_ext}-error.log", "w") as f:
            f.write(str(e))

        return
    finally:
        os.chdir(olddir)


def testing():
    rpe_dist_fn = "C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\src\\population_analysis\\plotting\\distance\\mlati10-2023-07-06-output.hdf-nwb-Euclidian-1.pickle"
    with open(rpe_dist_fn, "rb") as f:
        rpe = pickle.load(f)
    rpp_rpe_dist_rn = "C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\src\\population_analysis\\plotting\\distance\\0,100-dists-Euclidian-mlati10-2023-07-06-output.hdf-nwb-dir1.pickle"
    with open(rpp_rpe_dist_rn, "rb") as f:
        dist = pickle.load(f)

    plt.plot(dist, color="blue")
    plt.plot(np.mean(rpe, axis=0), color="orange")
    plt.show()

    tw = 2


def dist_compare():
    rpe_dist_fn = "C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\src\\population_analysis\\plotting\\distance\\mlati10-2023-07-06-output.hdf-nwb-Euclidian1.pickle"
    with open(rpe_dist_fn, "rb") as f:
        rpe = pickle.load(f)

    rpp_rpe_dist_rn = "C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\src\\population_analysis\\plotting\\distance\\new-mlati10-2023-07-06-output.hdf-nwb-Euclidian1.pickle"
    with open(rpp_rpe_dist_rn, "rb") as f:
        dist = pickle.load(f)

    plt.plot(np.mean(dist, axis=0), color="blue")
    plt.plot(np.mean(rpe, axis=0), color="orange")
    plt.show()

    tw = 2


def rpp_distance(sess: NWBSession):
    print("Initializing unit filters..")
    ufilt = sess.unit_filter_premade()
    ufilt.idxs()

    motdir = 1

    rpp_trial_filt = sess.trial_filter_rp_peri(0.4, 0.5, sess.trial_motion_filter(motdir))
    rpe_trial_filt = sess.trial_filter_rp_extra().append(sess.trial_motion_filter(motdir))

    rpp = sess.rp_peri_units()[ufilt.idxs()][:, rpp_trial_filt.idxs()]
    rpe = sess.units()[ufilt.idxs()][:, rpe_trial_filt.idxs()]
    print("Calculating dists..")

    quan = EuclidianQuantification()
    dists = []
    for t in range(35):  # (units, trials, t) -> (units, trials)
        dists.append(quan.calculate(rpp[:, :, t], rpe[:, :, t]))  # x
    plt.plot(dists, color="blue")

    baseline_fn = "mlati10-2023-07-06-output.hdf-nwb-Euclidian1.pickle"
    with open(baseline_fn, "rb") as f:
        baseline_dist = pickle.load(f)
    plt.plot(np.mean(baseline_dist, axis=0), color="orange")
    plt.show()
    tw = 2


# def baseline_plots(baseline_fn):
#     fig, ax = plt.subplots()
#     with open(baseline_fn, "rb") as f:
#         dists = pickle.load(f)
#     ax.plot()


def main():
    # print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    # grp = NWBSessionGroup("F:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    # filename, sess = next(grp.session_iter())
    # debug_rpe_baseline(sess, False)
    # rpp_distance(sess)
    # dist_compare()
    # testing()

    grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs")
    # grp = NWBSessionGroup("C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\scripts\\nwbs\\mlati7-2023-05-15-output")
    for folder, filename in grp.session_names_iter():
        debug_rpe_baseline(folder, filename, True)


if __name__ == "__main__":
    main()
