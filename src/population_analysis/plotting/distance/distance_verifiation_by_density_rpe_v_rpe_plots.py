import os.path

import numpy as np
import matplotlib.pyplot as plt
import pickle

from population_analysis.processors.filters import BasicFilter
from population_analysis.quantification import QuanDistribution
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_distance_density(data1, name1, data2, name2, quan, shuffle):
    # data1/2 is (units, trials, t)

    if shuffle:
        shuf = np.hstack([data1, data2]).swapaxes(0, 1)
        np.random.shuffle(shuf)
        shuf = shuf.swapaxes(0, 1)
        data1 = shuf[:, :data1.shape[1], :]
        data2 = shuf[:, data1.shape[1]:, :]

    num_t = data1.shape[2]
    dists = []
    for t in range(num_t):
        dists.append(
            quan.calculate(
                data1[:, :, t],
                data2[:, :, t]
            )
        )

    fig, ax = plt.subplots()
    ax.plot(dists)
    plt.show()


def plot_verif_rpe_v_rpe(sess: NWBSession, used_cached=True, suppress_plot=False):
    motdata = {}  # {1: <arr like (samples10k, 35), -1: ..}

    for motdir in [-1, 1]:
        pickle_fn = f"quan_dist_rpe_v_rpe_mot_{motdir}.pickle"
        if used_cached and os.path.exists(pickle_fn):
            with open(pickle_fn, "rb") as fff:
                motdata[motdir] = pickle.load(fff)
                continue
        print(f"Loading unit filter and trial filter for mot={motdir}..")
        # ufilt = sess.unit_filter_premade()
        ufilt = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.num_units)
        trial_filter = sess.trial_filter_rp_extra().append(sess.trial_motion_filter(motdir))
        quan = EuclidianQuantification()
        print("Calculating unit idxs and filter idxs..")
        units = sess.units()[ufilt.idxs()][:, trial_filter.idxs()]
        half_num_trials = int(units.shape[1]/2)

        # plot_distance_density(
        #     units[:, :half_num_trials], "RpExtra1",
        #     units[:, half_num_trials:], "RpExtra2",
        #     quan,
        #     True
        # )
        print("Starting Quantity Distribution calculations..")
        quan_dist = QuanDistribution(
            # units[:, :half_num_trials],
            # units[:, half_num_trials:],
            units[:, :40],
            units[:, 40:],
            quan
        )

        results = quan_dist.calculate()  # (10000, 35) arr of the distances
        motdata[motdir] = results
        print(f"Writing to pickle file '{pickle_fn}'")
        fp = open(pickle_fn, "wb")
        pickle.dump(results, fp)
        fp.close()

    if not suppress_plot:
        plt.title("RpExtra v RpExtra distance, 10k bootstrap mean")
        # plt.errorbar(range(35), np.mean(motdata[1], axis=0), label="1", yerr=np.std(motdata[1], axis=0))
        plt.errorbar(range(35), np.mean(motdata[-1], axis=0), label="-1", yerr=np.std(motdata[-1], axis=0), ecolor="oran")
        plt.xlabel("Time (20ms bins)")
        plt.ylabel("Euclidian distance")
        plt.legend()
        plt.show()

    return motdata
    tw = 2


def main():
    filename = "new_test"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")
    plot_verif_rpe_v_rpe(sess)
    # plot_verif_rpe_v_rpe(sess, False)
    tw = 2


if __name__ == "__main__":
    main()
