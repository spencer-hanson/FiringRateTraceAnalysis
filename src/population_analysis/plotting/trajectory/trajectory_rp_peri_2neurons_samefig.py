import numpy as np

from population_analysis.processors.nwb import NWBSession
from population_analysis.processors.nwb.filters import BasicFilter
import matplotlib.pyplot as plt


def plot_trajectory(u1, u2, ax, linelabel, linecolor="blue", startcolor="green", endcolor="red", startpoint_name="Startpoint", endpoint_name="Endpoint", traj=True, pts=True):
    if traj:
        ax.plot(u1, u2, color=linecolor, label=linelabel)

    if pts:
        ax.scatter(u1[0], u2[0], marker="o", color=startcolor, s=48, label=startpoint_name, zorder=5)
        ax.scatter(u1[-1], u2[-1], marker="o", color=endcolor, s=48, label=endpoint_name, zorder=5)
    # for i in range(1, len(u1)):
    #     x, y = u1[i - 1], u2[i - 1]
    #     dx = u1[i] - x
    #     dy = u2[i] - y
    #     rel_len = np.sqrt(dx * dx + dy * dy)
    #     ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=.02 * rel_len, head_length=.05 * rel_len,
    #              overhang=1, linestyle="dotted")


def plot_rp_peri_2neuron_samefig_trajectory(sess: NWBSession, ufilt):
    fig, axs = plt.subplots(2, 1)

    for idx, motdir in enumerate([-1, 1]):
        rp_peri_trial_filter = sess.trial_filter_rp_peri(sess.trial_motion_filter(motdir))
        neuron_trial_filter = BasicFilter(sess.probe_trial_idxs, sess.num_trials).append(
            sess.trial_motion_filter(motdir)
        )

        regular_units = np.mean(sess.units()[ufilt.idxs()][:, neuron_trial_filter.idxs()], axis=1)
        plot_trajectory(regular_units[0], regular_units[1], axs[idx], "RpExtra", "blue", "green", "red", "RpExtraStart", "RpExtraEnd", pts=False)

        rp_peri_units = np.mean(sess.rp_peri_units()[ufilt.idxs()][:, rp_peri_trial_filter.idxs()], axis=1)
        plot_trajectory(rp_peri_units[0], rp_peri_units[1], axs[idx], "RpPeri", "orange", "purple", "yellow", "RpPeriStart", "RpPeriEnd", pts=False)

        plot_trajectory(rp_peri_units[0], rp_peri_units[1], axs[idx], "", "orange", "purple", "yellow", "RpPeriStart", "RpPeriEnd", traj=False)
        plot_trajectory(regular_units[0], regular_units[1], axs[idx], "", "blue", "green", "red", "RpExtraStart", "RpExtraEnd", traj=False)

        axs[idx].legend()

    axs[1].set_xlabel(f"Unit 373 activity")
    axs[0].set_ylabel(f"Unit 233 activity motion=-1")
    axs[1].set_ylabel(f"Unit 233 activity motion=1")
    axs[0].set_title("RpExtra and RpPeri")

    plt.show()


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")

    ufilt = BasicFilter([373, 233], sess.units().shape[1])
    plot_rp_peri_2neuron_samefig_trajectory(sess, ufilt)

    # ufilt = sess.unit_filter_qm().append(
    #     sess.unit_filter_probe_zeta().append(
    #         sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
    #     )
    # )


if __name__ == "__main__":
    main()
