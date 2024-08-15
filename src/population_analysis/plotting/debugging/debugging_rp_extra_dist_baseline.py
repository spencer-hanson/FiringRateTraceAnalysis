import pickle
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import calc_quandist
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup


def debug_rpe_baseline(sess: NWBSession, use_cached):
    try:
        ufilt = sess.unit_filter_premade()
        quan = EuclidianQuantification()
        rpperi = sess.rp_peri_units().shape[1]
        rpextra = len(sess.trial_filter_rp_extra().idxs())
        prop = rpperi / rpextra
        quan_dist_motdir_dict = calc_quandist(sess, ufilt, sess.trial_filter_rp_extra(), sess.filename_no_ext, quan=quan, use_cached=use_cached, base_prop=prop)

    except Exception as e:
        raise e
        print(f"Error in session '{sess.filename_no_ext}' Error: '{str(e)}'")
        return


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



def main():
    # print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    # grp = NWBSessionGroup("E:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    # filename, sess = next(grp.session_iter())
    # debug_rpe_baseline(sess, False)
    dist_compare()
    # testing()

    # grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs")
    # for filename, sess in grp.session_iter():
    #     debug_rpe_baseline(sess, False)


if __name__ == "__main__":
    main()
