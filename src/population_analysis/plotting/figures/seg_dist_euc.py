import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import get_xaxis_vals
from population_analysis.processors.filters.trial_filters.rp_peri import RelativeTrialFilter
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_segment(rpperi_units, rpextra_units, title):
    quan = EuclidianQuantification()
    dists = []
    for t in range(NUM_FIRINGRATE_SAMPLES):
        dists.append(quan.calculate(rpextra_units[:, :, t], rpperi_units[:, :, t]))

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(get_xaxis_vals(), dists)
    axs[0].set_title(title)
    [axs[1].plot(get_xaxis_vals(), x) for x in np.mean(rpperi_units, axis=1)]
    plt.savefig(f"{title}.png")
    plt.show()


def segmented_rpe_rpp(sess):
    ufilt = sess.unit_filter_premade()

    rpperi = sess.rp_peri_units()[ufilt.idxs()]
    rpextra = sess.units()[ufilt.idxs()][:, sess.trial_filter_rp_extra().idxs()]

    mixed_rel_timestamps = sess.nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]
    mixed_trial_idxs = sess.nwb.processing["behavior"]["unit-trial-mixed"].data[:]
    stuff = []

    for i in range(10):
        st = (i-5)/10
        end = ((i-5)/10)+.1
        lt = mixed_rel_timestamps >= st
        gt = mixed_rel_timestamps <= end
        andd = np.logical_and(lt, gt)

        st = round(st, 3)
        end = round(end, 3)
        plot_segment(rpperi[:, andd], rpextra, f"num{i}_{st} to {end} RpPeri")

    tw = 2

    pass


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    nwbfiles = glob.glob("../../../../scripts/*/*-04-14*.nwb")
    # nwbfiles = glob.glob("../../../../scripts/*/*generated*.nwb")
    nwb_filename = nwbfiles[0]

    filepath = os.path.dirname(nwb_filename)
    filename = os.path.basename(nwb_filename)[:-len(".nwb")]

    sess = NWBSession(filepath, filename)
    segmented_rpe_rpp(sess)


if __name__ == "__main__":
    main()

