from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
import matplotlib.pyplot as plt

from population_analysis.processors.filters import BasicFilter
from population_analysis.processors.filters.trial_filters.rp_peri import RelativeTrialFilter
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.quantification.magnitude_difference import MagDiffQuantification


def main():
    filename = "new_test"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")

    # ufilt = sess.unit_filter_qm().append(
    #     sess.unit_filter_probe_zeta().append(
    #         sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
    #     )
    # )

    # ufilt = BasicFilter([373, 233], sess.units().shape[1])
    ufilt = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.units().shape[1])

    rp_extra = sess.units()[ufilt.idxs()]
    rp_peri = sess.rp_peri_units()[ufilt.idxs()]

    quantification_list = [
        # MagQuoQuantification(),
        EuclidianQuantification(),
        MagDiffQuantification(),
    ]

    fig, axs = plt.subplots(nrows=len(quantification_list), ncols=2, squeeze=False, sharex=True)  # 2 cols for motion dirs
    fig.tight_layout()

    for col_idx, motdir in enumerate([-1, 1]):
        rp_e_filter = sess.trial_motion_filter(motdir).append(BasicFilter(sess.probe_trial_idxs, rp_extra.shape[1]))
        rp_p_filter = RelativeTrialFilter(sess.trial_motion_filter(motdir), sess.mixed_trial_idxs)

        rpe = rp_extra[:, rp_e_filter.idxs()]
        rpp = rp_peri[:, rp_p_filter.idxs()]

        for row_idx, quan in enumerate(quantification_list):
            axis = axs[row_idx, col_idx]
            dist_arr = []
            for t in range(NUM_FIRINGRATE_SAMPLES):
                dist_arr.append(quan.calculate(rpp[:, :, t], rpe[:, :, t]))
            axis.plot(dist_arr)
            tw = 2
            axis.title.set_text(f"Rpp vs Rpe {quan.get_name()} - Motion={motdir}")

            if row_idx == len(quantification_list) - 1:
                axis.set_xlabel("Time (20 ms bins)")

    plt.show()
    tw = 2


if __name__ == "__main__":
    main()

