import numpy as np

from population_analysis.processors.nwb import NWBSession
from population_analysis.processors.nwb.filters import BasicFilter
import matplotlib.pyplot as plt


def plot_trajectory_arrows(u1, u2, ax):
    ax.scatter(u1[0], u2[0], marker="o", color="green", s=64, label="Startpoint")
    ax.scatter(u1[-1], u2[-1], marker="o", color="red", s=64, label="Endpoint")
    for i in range(1, len(u1)):
        x, y = u1[i - 1], u2[i - 1]
        dx = u1[i] - x
        dy = u2[i] - y
        rel_len = np.sqrt(dx * dx + dy * dy)
        ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=.02 * rel_len, head_length=.05 * rel_len,
                 overhang=1, linestyle="dotted")


def plot_rp_peri_2neuron_trajectory(sess: NWBSession, ufilt):
    fig, axs = plt.subplots(2, 2)

    for idx, motdir in enumerate([-1, 1]):
        rp_peri_trial_filter = sess.trial_filter_rp_peri(sess.trial_motion_filter(motdir))
        neuron_trial_filter = BasicFilter(sess.probe_trial_idxs, sess.num_trials).append(
            sess.trial_motion_filter(motdir)
        )

        regular_units = np.mean(sess.units()[ufilt.idxs()][:, neuron_trial_filter.idxs()], axis=1)
        plot_trajectory_arrows(regular_units[0], regular_units[1], axs[idx, 0])
        axs[idx, 0].legend()

        rp_peri_units = np.mean(sess.rp_peri_units()[ufilt.idxs()][:, rp_peri_trial_filter.idxs()], axis=1)
        plot_trajectory_arrows(rp_peri_units[0], rp_peri_units[1], axs[idx, 1])
        axs[idx, 1].legend()

        if idx == 1:
            axs[idx, 0].set_xlabel(f"Unit 373 activity")

        # axs[1].set_xlabel("Unit 233 activity")
        # axs[1].set_ylabel("Unit 273 activity motion=1")
    axs[1, 0].set_ylabel(f"Unit 233 activity motion={motdir}")
    axs[0, 0].set_title("Extrasaccadic - RpExtra")
    axs[0, 1].set_title("Perisaccadic - RpPeri")

    plt.show()


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")

    ufilt = BasicFilter([373, 233], sess.units().shape[1])
    plot_rp_peri_2neuron_trajectory(sess, ufilt)

    # ufilt = sess.unit_filter_qm().append(
    #     sess.unit_filter_probe_zeta().append(
    #         sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
    #     )
    # )


if __name__ == "__main__":
    main()
