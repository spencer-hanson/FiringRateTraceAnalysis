import numpy as np

from population_analysis.processors.filters import BasicFilter
from population_analysis.sessions.saccadic_modulation import NWBSession
import matplotlib.pyplot as plt


def plot_2neuron_trajectory(sess, ufilt):
    fig, main_axs = plt.subplots(2, 2)

    neurons = [
        BasicFilter(sess.probe_trial_idxs, sess.num_trials),
        BasicFilter(sess.mixed_trial_idxs, sess.num_trials)
    ]

    for idx, resp_trial_filter in enumerate(neurons):
        axs = main_axs[:, idx]
        for motdir, ax in zip([-1, 1], axs):
            trial_filt = sess.trial_motion_filter(motdir).append(resp_trial_filter)
            units = np.mean(sess.units()[ufilt.idxs()][:, trial_filt.idxs()], axis=1)
            u1 = units[0]
            u2 = units[1]

            ax.scatter(u1[0], u2[0], marker="o", color="green", s=64, label="Startpoint")
            ax.scatter(u1[-1], u2[-1], marker="o", color="red", s=64, label="Endpoint")
            for i in range(1, len(u1)):
                x, y = u1[i-1], u2[i-1]
                dx = u1[i]-x
                dy = u2[i]-y
                rel_len = np.sqrt(dx*dx + dy*dy)
                ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=.02*rel_len, head_length=.05*rel_len, overhang=1, linestyle="dotted")
                # ax.annotate("", xy=(x, y), xytext=(dx, dy), arrowprops=dict(arrowstyle="->"))
            # ax.plot(u1, u2)
            if idx == 0:
                ax.set_ylabel(f"motion={motdir}")
            ax.legend()

        axs[1].set_xlabel("Unit 373 activity")

    main_axs[1, 0].set_ylabel("Unit 233 activity motion=1")
    main_axs[0, 0].set_title("Extrasaccadic - RpExtra")
    main_axs[0, 1].set_title("Perisaccadic - Rmixed")

    plt.show()
    pass


def main():
    filename = "new_test"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")

    ufilt = BasicFilter([373, 233], sess.units().shape[1])
    plot_2neuron_trajectory(sess, ufilt)

    # ufilt = sess.unit_filter_qm().append(
    #     sess.unit_filter_probe_zeta().append(
    #         sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
    #     )
    # )


if __name__ == "__main__":
    main()
