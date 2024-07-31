import math

import numpy as np
from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import plot_verif_rpe_v_rpe, \
    calc_quandist
from population_analysis.processors.filters import BasicFilter
from population_analysis.processors.filters.trial_filters.rp_peri import RelativeTrialFilter
from population_analysis.quantification.angle import AngleQuantification
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession
import matplotlib.pyplot as plt
import scipy.stats as st
from distance_rpp_rpe_errorbars_plots import confidence_interval


def calc_conf(quan_dist_data, confidence_val):
    means = []
    uppers = []
    lowers = []
    for t in range(NUM_FIRINGRATE_SAMPLES):
        lower, upper = confidence_interval(quan_dist_data[:, t], confidence_val)
        mean = np.mean(quan_dist_data[:, t], axis=0)
        means.append(mean)
        uppers.append(upper)
        lowers.append(lower)
    return means, uppers, lowers


def rpp_rpe_rs_errorbars(sess: NWBSession, quan, quan_dist_motdir_dict, confidence_val, ufilt):
    # quan_dist_motdir_dict is a dict with the keys as 1 or -1 and the data as (10k, 35) for the quan distribution
    fig, axs = plt.subplots(ncols=2)
    # fig.tight_layout()

    rp_extra = sess.units()[ufilt.idxs()]
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]
    rs = sess.units()[ufilt.idxs()]

    for col_idx, motdir in enumerate([-1, 1]):
        ax = axs[col_idx]
        rp_e_filter = sess.trial_motion_filter(motdir).append(BasicFilter(sess.probe_trial_idxs, rp_extra.shape[1]))
        rp_p_filter = RelativeTrialFilter(sess.trial_motion_filter(motdir), sess.mixed_trial_idxs)
        rs_filter = sess.trial_motion_filter(motdir).append(BasicFilter(sess.saccade_trial_idxs, rs.shape[1]))

        rpe = rp_extra[:, rp_e_filter.idxs()]
        rpp = rp_peri[:, rp_p_filter.idxs()]
        rss = rs[:, rs_filter.idxs()]

        dist_arr = []
        rs_dist_arr = []

        for t in range(NUM_FIRINGRATE_SAMPLES):
            dist_arr.append(quan.calculate(rpp[:, :, t], rpe[:, :, t]))
            rs_dist_arr.append(quan.calculate(rpp[:, :, t], rss[:, :, t]))

        quan_dist_data = quan_dist_motdir_dict[motdir]

        # rpp
        ax.plot(dist_arr, label="RpPeri")
        # rs
        ax.plot(rs_dist_arr, color="green", label="Rs")

        means, uppers, lowers = calc_conf(quan_dist_data, confidence_val)

        # rp peri
        ax.plot(means, color="orange", label="RpPeri")
        ax.plot(uppers, color="orange", linestyle="dotted")
        ax.plot(lowers, color="orange", linestyle="dotted")

        # np.mean(quan_dist_data, axis=0)
        # ax.plot(means, color="orange")

        tw = 2
        ax.title.set_text(f"RpP vs RpE & Rs {quan.get_name()} - Motion={motdir}")
        ax.set_xlabel("Time (20 ms bins)")
        ax.legend()

    plt.show()
    tw = 2


def main():
    confidence = 0.95
    month = "05"
    day = 19
    year = 2023
    mouse_name = "mlati7"
    filename = f"{mouse_name}-{year}-{month}-{day}-output"
    filepath = f"../../../../scripts/{filename}"
    filename = f"{filename}.hdf-nwb"

    sess = NWBSession(filepath, filename, "../graphs")
    # ufilt = BasicFilter([231, 235], sess.num_units)  # 05-19-2023
    ufilt = sess.unit_filter_premade()

    # for 05-15
    # ufilt = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.num_units)  # 05-15-2023
    # ufilt = BasicFilter([231, 235], sess.num_units)  # 05-19-2023
    # ufilt = BasicFilter([244, 365], sess.num_units)
    # ufilt = sess.unit_filter_premade()
    # ufilt = BasicFilter.empty(sess.num_units)
    # use_cached = False
    use_cached = True

    # quan = EuclidianQuantification()
    quan = AngleQuantification()
    quan_dist_motdir_dict = calc_quandist(sess, ufilt, sess.trial_filter_rp_extra(), "rpp_rpe_rs", quan=quan, use_cached=use_cached)
    rpp_rpe_rs_errorbars(sess, quan, quan_dist_motdir_dict, confidence, ufilt)

    # confidence_interval(quan_dist_motdir_dict[-1][:, 0], confidence, plot=True)  # plot first timepoints CDF for 95% conf interval
    tw = 2


if __name__ == "__main__":
    main()
