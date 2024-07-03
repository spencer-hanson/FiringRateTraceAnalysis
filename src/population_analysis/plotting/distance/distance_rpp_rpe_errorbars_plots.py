import math

import numpy as np
from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import plot_verif_rpe_v_rpe
from population_analysis.processors.filters import BasicFilter
from population_analysis.processors.filters.trial_filters.rp_peri import RelativeTrialFilter
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession
import matplotlib.pyplot as plt
import scipy.stats as st


def mean_confidence_interval(data, confidence=0.95):
    samp_size = data.shape[0]
    std = np.std(data)

    # a = 1.0 * np.array(data)
    # n = len(a)
    # m, se = np.mean(a), st.sem(a)
    # h = se * st.t.ppf((1 + confidence) / 2., n-1)
    # return m, h
    z = st.norm.ppf(q=0.00005)
    err = z * (std/math.sqrt(samp_size))
    return np.mean(data), np.abs(err)


def rpp_rpe_errorbars(sess: NWBSession, quan, quan_dist_motdir_dict):
    # quan_dist_motdir_dict is a dict with the keys as 1 or -1 and the data as (10k, 35) for the quan distribution
    fig, axs = plt.subplots(ncols=2)
    fig.tight_layout()

    ufilt = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.num_units)

    rp_extra = sess.units()[ufilt.idxs()]
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]

    for col_idx, motdir in enumerate([-1, 1]):
        ax = axs[col_idx]
        rp_e_filter = sess.trial_motion_filter(motdir).append(BasicFilter(sess.probe_trial_idxs, rp_extra.shape[1]))
        rp_p_filter = RelativeTrialFilter(sess.trial_motion_filter(motdir), sess.mixed_trial_idxs)

        rpe = rp_extra[:, rp_e_filter.idxs()]
        rpp = rp_peri[:, rp_p_filter.idxs()]

        dist_arr = []
        for t in range(NUM_FIRINGRATE_SAMPLES):
            dist_arr.append(quan.calculate(rpp[:, :, t], rpe[:, :, t]))
        quan_dist_data = quan_dist_motdir_dict[motdir]
        # ax.plot(dist_arr)

        confdiffs = []
        means = []
        for t in range(NUM_FIRINGRATE_SAMPLES):
            mean, diff = mean_confidence_interval(quan_dist_data[:, t])
            confdiffs.append(diff)
            means.append(mean)
        ax.errorbar(range(35), np.mean(quan_dist_data, axis=0), yerr=confdiffs, ecolor="red")
        # ax.plot(means, color="orange")

        tw = 2
        ax.title.set_text(f"Rpp vs Rpe {quan.get_name()} - Motion={motdir}")
        ax.set_xlabel("Time (20 ms bins)")

    plt.show()
    tw = 2


def main():
    filename = "new_test"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")
    quan_dist_motdir_dict = plot_verif_rpe_v_rpe(sess, suppress_plot=True)
    rpp_rpe_errorbars(sess, EuclidianQuantification(), quan_dist_motdir_dict)

    # plot_verif_rpe_v_rpe(sess, False)
    tw = 2


if __name__ == "__main__":
    main()
