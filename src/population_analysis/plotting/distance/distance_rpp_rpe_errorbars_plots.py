import math
import os
import pickle
from typing import Union

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

from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup


def confidence_interval(data, confidence_val, plot=False):
    # data is an arr (10k,)
    hist = np.histogram(data, bins=200)

    pdf = hist[0] / sum(hist[0])
    cdf = np.cumsum(pdf)

    lower_idx = np.where(cdf > 1 - confidence_val)[0][0]
    lower = hist[1][lower_idx + 1]

    upper_idx = np.where(cdf > confidence_val)[0][0]
    upper = hist[1][upper_idx + 1]

    if plot:
        plt.plot(hist[1][1:], cdf)
        plt.vlines(lower, 0, 1.0, color="red")
        plt.vlines(upper, 0, 1.0, color="red")
        plt.show()
    return lower, upper


def get_xaxis_vals():
    return np.arange(35) * 20 - 200


def distance_errorbars(ax, units1, units2, quan, quandist_dict, motdir, confidence_val, save_dists: Union[bool, str] = False):
    dist_arr = []
    for t in range(NUM_FIRINGRATE_SAMPLES):
        dist_arr.append(quan.calculate(units1[:, :, t], units2[:, :, t]))

    if save_dists:
        with open(save_dists, "wb") as f:
            pickle.dump(dist_arr, f)

    ax.plot(get_xaxis_vals(), dist_arr)

    if motdir == 0:  # Both dirs
        quan_dist_data = np.vstack([quandist_dict[-1], quandist_dict[1]])
    else:
        quan_dist_data = quandist_dict[motdir]

    means = []
    uppers = []
    lowers = []
    for t in range(NUM_FIRINGRATE_SAMPLES):
        lower, upper = confidence_interval(quan_dist_data[:, t], confidence_val)
        mean = np.mean(quan_dist_data[:, t], axis=0)
        means.append(mean)
        uppers.append(upper)
        lowers.append(lower)

    ax.plot(get_xaxis_vals(), means, color="orange")
    ax.plot(get_xaxis_vals(), uppers, color="orange", linestyle="dotted")
    ax.plot(get_xaxis_vals(), lowers, color="orange", linestyle="dotted")

    ax.title.set_text(f"{quan.get_name()} {motdir}")
    ax.set_xlabel("Time (ms)")

    return dist_arr, means, uppers, lowers

def rpp_rpe_errorbars(sess: NWBSession, quans: list, confidence_val, ufilt, cache_filename, save_filepath=None, use_cached=False):
    # quan_dist_motdir_dict is a dict with the keys as 1 or -1 and the data as (10k, 35) for the quan distribution
    fig, axs = plt.subplots(ncols=2, nrows=len(quans))

    rp_extra = sess.units()[ufilt.idxs()]
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]

    for quan_idx in range(len(quans)):
        quan = quans[quan_idx]
        quan_dist_motdir_dict = calc_quandist(sess, ufilt, sess.trial_filter_rp_extra(), cache_filename, quan=quan, use_cached=use_cached)

        for col_idx, motdir in enumerate([-1, 1]):
            distance_errorbars(
                axs[quan_idx, col_idx],
                rp_extra[:, sess.trial_motion_filter(motdir).append(sess.trial_filter_rp_extra()).idxs()],
                rp_peri[:, sess.trial_filter_rp_peri(sess.trial_motion_filter(motdir)).idxs()],
                quan,
                quan_dist_motdir_dict,
                motdir,
                confidence_val,
                save_dists=f"dists-{cache_filename}-{quan.get_name()}{motdir}.pickle"
            )

    if save_filepath is None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    tw = 2


def main():
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening

    confidence = 0.95

    print("Loading group..")
    grp = NWBSessionGroup("../../../../scripts")
    for name, sess in grp.session_iter():  # TODO Theres a memory leak somewhere in this shit, fuck python
        use_cached = False
        # use_cached = True

        quans = [
            EuclidianQuantification(),
            AngleQuantification()
        ]

        foldername = "rpp_rpe_errorbars_plots"
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        save_filename = os.path.join(foldername, name + ".png")
        if os.path.exists(save_filename):
            print(f"Image already rendered '{save_filename}' skipping..")
            continue

        print(f"Processing '{name}'..")
        ufilt = sess.unit_filter_premade()
        try:
            rpp_rpe_errorbars(sess, quans, confidence, ufilt, name, save_filepath=save_filename, use_cached=True)
        except Exception as e:
            print(f"ERROR PROCESSING FILE '{name}' Skipping! Error: '{str(e)}'")
            continue

    # date = "2023-05-19"
    # filepath = f"../../../../scripts/mlati7-{date}-output"
    # filename = f"mlati7-{date}-output.hdf-nwb"
    # sess = NWBSession(filepath, filename, "../graphs")
    #

    # for 05-15
    # ufilt = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.num_units)  # 05-15-2023
    # ufilt = BasicFilter([231, 235], sess.num_units)  # 05-19-2023
    # ufilt = BasicFilter([244, 365], sess.num_units)
    # ufilt = sess.unit_filter_premade()
    # ufilt = BasicFilter.empty(sess.num_units)
    # ufilt = BasicFilter([231, 235], sess.num_units)  # 05-19-2023

    # confidence_interval(quan_dist_motdir_dict[-1][:, 0], confidence, plot=True)  # plot first timepoints CDF for 95% conf interval
    tw = 2


if __name__ == "__main__":
    main()
