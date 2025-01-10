import matplotlib.pyplot as plt
import numpy as np

from population_analysis.sessions.saccadic_modulation import NWBSession


def main():
    # filename = "not_smoothed_2023-05-15_mlati7_output"
    filename = "new_test"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../../../../graphs")

    unit_num = 244
    # unit_num = 233

    for motdir in [-1, 1]:
        filt = sess.trial_motion_filter(motdir)
        filt.append(sess.trial_filter_rmixed())

        spikes = sess.spikes()[:, filt.idxs()]
        unit_spikes = spikes[unit_num]

        def do_plot(d):
            plt.plot(d)
            plt.show()

        plt.title(f"Number of spikes per trial of unit {unit_num} mot={motdir}")
        plt.plot(np.sum(unit_spikes, axis=1))
        plt.hlines(0, 0, unit_spikes.shape[0], color="red")
        plt.show()

        plt.title(f"Mini spikes raster of unit {unit_num} mot={motdir}")
        [plt.plot(s, color="blue", alpha=.1) for s in unit_spikes[:50]]  # TODO only first 50 trials?
        plt.show()

        plt.title(f"Mini mean responses of unit {unit_num} mot={motdir}")
        [plt.plot(w, color="blue", alpha=.1) for w in sess.units()[:, filt.idxs()][unit_num][:100]]
        plt.show()


    tw = 2


if __name__ == "__main__":
    main()

