import numpy as np
from matplotlib import pyplot as plt

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.processors.nwb import NWBSession
from population_analysis.processors.nwb.filters.__init__ import BasicFilter
from population_analysis.processors.nwb.filters.trial_filters.probe_offset import ProbeOffsetTrialFilter
from population_analysis.processors.nwb.filters.trial_filters.rp_peri import RelativeTrialFilter
from population_analysis.quantification.euclidian import EuclidianQuantification


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../scripts", filename, "../graphs", filter_mixed=False)

    ufilt = sess.unit_filter_qm().append(
        sess.unit_filter_probe_zeta().append(
            sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
        )
    )

    bins = np.arange(-.2, .2, .01)
    units = sess.units()
    quan = EuclidianQuantification()

    fig, axs = plt.subplots(nrows=NUM_FIRINGRATE_SAMPLES, ncols=2, figsize=(8, 4*NUM_FIRINGRATE_SAMPLES))
    fig.subplots_adjust(wspace=0.2, hspace=.8)
    # fig.tight_layout()
    max_val = 0
    mot_dist_from_time = {}  # dist_from_time but indexed by the motion direction

    for mot_idx, motdir in enumerate([-1, 1]):
        dist_from_time = {x: [] for x in range(NUM_FIRINGRATE_SAMPLES)}  # dict like {t: [bin1, bin2, ..], ..} where bin1 is the distance from rp_peri to rp_extra in bin1, and t is the time to take the population at
        for bin_num in range(len(bins)):
            rp_peri_filt = RelativeTrialFilter(sess.trial_motion_filter(motdir), sess.mixed_trial_idxs).append(
                ProbeOffsetTrialFilter(sess.mixed_rel_timestamps, bins, bin_num + 1)  # bins start at 1
            )
            rp_extra_filt = sess.trial_motion_filter(motdir).append(  # Filter by mot dir
                BasicFilter(sess.mixed_trial_idxs, units.shape[1])  # Filter by mixed
            )

            rp_peri = sess.rp_peri_units()[ufilt.idxs()][:, rp_peri_filt.idxs()]
            rp_extra = units[ufilt.idxs()][:, rp_extra_filt.idxs()]
            for t in range(NUM_FIRINGRATE_SAMPLES):
                dist = quan.calculate(
                    rp_peri[:, :, t],
                    rp_extra[:, :, t]
                )
                if dist > max_val:
                    max_val = dist

                dist_from_time[t].append(dist)
        # get distribution of dists
        all_dists = []
        for t in range(NUM_FIRINGRATE_SAMPLES):
            all_dists.extend(dist_from_time[t])
        q75 = np.percentile(all_dists, 99)
        min_val = np.min(all_dists)
        mot_dist_from_time[motdir] = dist_from_time

        # After calculating all bin vals
        for t in range(NUM_FIRINGRATE_SAMPLES):
            ax = axs[t, mot_idx]
            ax.plot(bins, dist_from_time[t])
            if t == 0:
                ax.set_title(f"Motion = {motdir}, t = {t}")
            else:
                ax.set_title(f"t = {t}")
            ax.set_xlabel("ms from probe")
            ax.set_ylim([min_val,  q75])
            # ax.set_yticks([])
            # ax.axis([0, len(bins), 0, max_val])

    fig.suptitle("Euclidian Distance between RpPeri and RpExtra, binned by time from the Probe")
    fig.savefig("distance_probe_offset_binned_all.png")

    for motdir in [-1, 1]:
        hsv = plt.get_cmap("hsv")
        ax = plt.figure().add_subplot(projection='3d')
        all_data = np.array([v for k, v in mot_dist_from_time[motdir].items()])
        for t in range(NUM_FIRINGRATE_SAMPLES):
            z = all_data[t, :]
            x = bins
            y = [t]*len(bins)
            ax.plot(x, y, z, color=hsv(t))
        plt.show()
        tw = 2

        tw = 2
    tw = 2



if __name__ == "__main__":
    main()

