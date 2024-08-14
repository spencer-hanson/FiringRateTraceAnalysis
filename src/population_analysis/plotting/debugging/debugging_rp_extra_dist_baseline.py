from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import calc_quandist
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup


def debug_rpe_baseline(sess: NWBSession, use_cached):
    try:
        ufilt = sess.unit_filter_premade()
        quan = EuclidianQuantification()
        quan_dist_motdir_dict = calc_quandist(sess, ufilt, sess.trial_filter_rp_extra(), sess.filename_no_ext, quan=quan,
                                              use_cached=use_cached)
    except Exception as e:
        print(f"Error in session '{sess.filename_no_ext}' Error: '{str(e)}'")
        return


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs")
    confidence_val = 0.95

    for filename, sess in grp.session_iter():
        debug_rpe_baseline(sess, False)


if __name__ == "__main__":
    main()
