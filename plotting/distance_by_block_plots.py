import numpy as np

from population_analysis.processors.nwb import NWBSessionProcessor
from distance_plots import calc_dists
import matplotlib.pyplot as plt

from population_analysis.quantification.euclidian import EuclidianQuantification


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSessionProcessor("../scripts", filename, "../graphs")

    filt = sess.qm_unit_filter().append(
        sess.probe_zeta_unit_filter().append(
            sess.activity_threshold_unit_filter(5, .2, 1, 1, .9, .4)
        )
    )

    probe_units = sess.probe_units()[filt.idxs()]
    saccade_units = sess.saccade_units()[filt.idxs()]

    trial_block_idxs = sess.trial_block_idxs()
    trial_dirs = sess.trial_motion_directions()

    for motion_dir in [-1, 1]:
        unique_block_idxs = np.unique(trial_block_idxs)
        quan = EuclidianQuantification("DistanceByBlock")
        dists = []

        for block_idx in unique_block_idxs:
            block_trial_idxs = np.where(trial_block_idxs == block_idx)[0]
            mot_trial_idxs = np.where(trial_dirs[block_trial_idxs] == motion_dir)[0]
            if len(mot_trial_idxs) > 0:
                probe = np.mean(probe_units[:, mot_trial_idxs], axis=1)  # Average along trials
                sacc = np.mean(saccade_units[:, mot_trial_idxs], axis=1)
                dists.append(quan.calculate(probe, sacc))

        plt.suptitle(f"RpExtra - Rs Distance By Block Index for motion={motion_dir}")
        plt.xlabel("Block idx num")
        plt.ylabel("Distance value")
        plt.plot(dists)
        plt.show()
    tw = 2


if __name__ == "__main__":
    main()
