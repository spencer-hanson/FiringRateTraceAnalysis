import pickle

from matplotlib import pyplot as plt

from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import distance_errorbars
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.group import SessionGroup


def euclidian_distance_rpe_rpp(sess):
    cache = True
    fig, ax = plt.subplots()
    ufilt = sess.unit_filter_premade()

    rp_extra = sess.units()[ufilt.idxs()]
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]
    quan = EuclidianQuantification()

    rpperi = sess.rp_peri_units().shape[1]
    rpextra = len(sess.trial_filter_rp_extra().idxs())
    prop = rpperi / rpextra
    prop = prop / 10
    prop = prop / 2  # TODO motion direction ratio

    motions = [1]
    # quan_dist_motdir_dict = calc_quandist(sess, ufilt, sess.trial_filter_rp_extra(), "fig-eucdist", prop, quan=quan, use_cached=cache, motions=motions)

    fn = "C:\\Users\\denma\\Documents\\GitHub\\SaccadePopulationAnalysis\\src\\\population_analysis\\\plotting\\\debugging\\mlati10-2023-07-06-output.hdf-nwb-Euclidian1.pickle"
    with open(fn, "rb") as f:
        quan_dist_motdir_dict = pickle.load(f)
    quan_dist_motdir_dict = {1: quan_dist_motdir_dict}
    dist_arr, rpe_means, rpe_uppers, rpe_lowers = distance_errorbars(
        ax,
        rp_extra,
        rp_peri,
        quan,
        quan_dist_motdir_dict,
        1,
        0.99,
        save_dists=f"dists-fig-eucdist.pickle"
    )

    # ax.vlines(0, 0, 75, color="black", linestyles="dashed")
    ax.vlines(0, 0, 2, color="black", linestyles="dashed")

    ax.set_title("RpExtra - RpPeri Euclidian Distance")
    ax.set_yticks([])

    plt.savefig("euclidian-distance.svg")
    plt.show()

    joshdata = {
        "rp_extra_lower_bound": rpe_lowers,
        "rp_extra_mean": rpe_means,
        "rp_extra_upper_bound": rpe_uppers,
        "rp_peri_v_rp_extra_euclidian_distance": dist_arr
    }

    print("Writing to file")
    with open("euclidian-distance-errorbars-data.pickle", "wb") as f:
        pickle.dump(joshdata, f)

    tw = 2


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    # nwbfiles = glob.glob("../../../../scripts/*/*-04-14*.nwb")
    # nwbfiles = glob.glob("C:\\Users\\Matrix\\Downloads\\tmp\\*04-14*.nwb")
    # nwbfiles = glob.glob("C:\\Users\\Matrix\\Downloads\\tmp\\*05-15*.nwb")
    grp = SessionGroup("F:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    _, sess = next(grp.session_iter())
    euclidian_distance_rpe_rpp(sess)

    # nwbfiles = glob.glob("../../../../scripts/*/*generated*.nwb")
    # nwb_filename = nwbfiles[0]
    # filepath = os.path.dirname(nwb_filename)
    # filename = os.path.basename(nwb_filename)[:-len(".nwb")]
    # sess = NWBSession(filepath, filename, mixed_probe_range=.1)  # For final figure want with 100ms range
    # euclidian_distance_rpe_rpp(sess)


if __name__ == "__main__":
    main()

