import matplotlib.pyplot as plt
import numpy as np

from population_analysis.processors.filters import BasicFilter
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_rp_peri_mean_responses(sess, unit_filter, ax_list=None):

    rp_peri = sess.rp_peri_units()[unit_filter.idxs()]
    neg_motfilt = RelativeTrialFilter(sess.trial_motion_filter(-1), sess.mixed_trial_idxs)
    pos_motfilt = RelativeTrialFilter(sess.trial_motion_filter(1), sess.mixed_trial_idxs)

    neg_mean_units = np.mean(rp_peri[:, neg_motfilt.idxs()], axis=1)
    pos_mean_units = np.mean(rp_peri[:, pos_motfilt.idxs()], axis=1)

    if ax_list is None:
        fig, axs = plt.subplots(2, 1)
        fig.subplots_adjust(wspace=0.2, hspace=.3)
    else:
        axs = ax_list

    for unit in neg_mean_units:
        axs[0].plot(unit)

    for unit in pos_mean_units:
        axs[1].plot(unit)

    axs[0].set_title("RpPeri mean responses Motion = -1")
    axs[1].set_title("Motion = 1")

    if ax_list is None:
        plt.show()
    tw = 2


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    # sess = NWBSession("../scripts", filename, "../graphs")
    sess = NWBSession("../../../../scripts", filename, "../../../../graphs", use_normalized_units=True)
    unit_filter = sess.unit_filter_qm().append(
        sess.unit_filter_probe_zeta().append(
            sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
        )
    )

    plot_rp_peri_mean_responses(sess, unit_filter)


if __name__ == "__main__":
    main()
