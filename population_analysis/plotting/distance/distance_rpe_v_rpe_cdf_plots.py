from population_analysis.sessions.saccadic_modulation import NWBSession


def main():
    filename = "new_test"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")

    quan_dist_motdir_dict = plot_verif_rpe_v_rpe(sess, True, suppress_plot=True)
    # quan_dist_motdir_dict = plot_verif_rpe_v_rpe(sess, False, suppress_plot=True)
    rpp_rpe_errorbars(sess, EuclidianQuantification(), quan_dist_motdir_dict)

    tw = 2


quan_dist_motdir_dict = plot_verif_rpe_v_rpe(sess, True, suppress_plot=True)