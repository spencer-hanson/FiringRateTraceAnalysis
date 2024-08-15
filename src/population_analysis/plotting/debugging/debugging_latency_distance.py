import glob
import os
import pickle
import shutil

import numpy as np
import matplotlib.pyplot as plt

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import confidence_interval, get_xaxis_vals
from population_analysis.plotting.figures.frac_sig_euc_dist_bars_max_vals import ensure_rpextra_exists
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup


def debug_latency_dists(sess, confidence_val, filename):
    sess_dir = os.path.join("latency_debug", f"{filename}-dir")

    if not os.path.exists(sess_dir):
        os.mkdir(sess_dir)

    save_fn_fmt = os.path.join(sess_dir, "{}.png")
    quan = EuclidianQuantification()
    motdir = 1

    rpextra_error_distribution_fn = f"{filename}-{quan.get_name()}{motdir}.pickle"
    os.chdir("../distance")

    rp_extra_exists, ex = ensure_rpextra_exists(rpextra_error_distribution_fn, sess, filename, quan)
    if not rp_extra_exists:
        print(f"Error calculating RpExtra distance distribution for '{filename}'!")
        raise ex
    mmax = 10

    allfig, allax = plt.subplots(ncols=mmax, sharey=True, sharex=True, figsize=(16, 4))

    for i in range(mmax):
        st = (i - (mmax / 2)) / 10
        end = ((i - (mmax / 2)) / 10) + .1
        rnd = lambda x: int(x * 1000)
        latency_key = f"{rnd(st)},{rnd(end)}"
        latency_dist_fn = f"{latency_key}-dists-{quan.get_name()}-{filename}-dir{motdir}.pickle"
        if os.path.exists(latency_dist_fn):
            print(f"Precalculated latency {latency_key} found..")
            with open(latency_dist_fn, "rb") as f:
                distances = pickle.load(f)

            with open(rpextra_error_distribution_fn, "rb") as f:
                rpextra_error_distribution = pickle.load(f)

            fig, oneax = plt.subplots()

            means = []
            uppers = []
            lowers = []
            for t in range(NUM_FIRINGRATE_SAMPLES):
                lower, upper = confidence_interval(rpextra_error_distribution[:, t], confidence_val)
                mean = np.mean(rpextra_error_distribution[:, t], axis=0)
                means.append(mean)
                uppers.append(upper)
                lowers.append(lower)
            for j, ax in enumerate([oneax, allax[i]]):
                ax.plot(get_xaxis_vals(), distances, color="blue")
                ax.plot(get_xaxis_vals(), means, color="orange")
                ax.plot(get_xaxis_vals(), uppers, color="orange", linestyle="dotted")
                ax.plot(get_xaxis_vals(), lowers, color="orange", linestyle="dotted")
                ax.title.set_text(latency_key)
                if i != 0 and j != 1:
                    ax.set_yticks([])

            save_fn = save_fn_fmt.format(latency_key)
            print(f"Saving {save_fn}")
            os.chdir("../debugging")
            fig.savefig(save_fn)
            os.chdir("../distance")
            plt.close(fig)
        else:
            raise ValueError("Distances not precalculated, use frac_sig_euc_dist_bars_max_vals.py")

    os.chdir("../debugging")
    print("Saving allfig..")
    allfig.savefig(save_fn_fmt.format("all"))
    tw = 2


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    # grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs")
    # grp = NWBSessionGroup("E:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    grp = NWBSessionGroup("../../../../scripts/mlati10*07-06*")
    confidence_val = 0.95
    if not os.path.exists("latency_debug"):
        os.mkdir("latency_debug")

    for filename, sess in grp.session_iter():
        debug_latency_dists(sess, confidence_val, filename)

    all_files = glob.glob("latency_debug/**/*all*.png")
    if not os.path.exists(os.path.join("latency_debug", "all")):
        os.mkdir(os.path.join("latency_debug", "all"))

    for fn in all_files:
        new_name = f"{os.path.basename(os.path.dirname(fn))}-all.png"
        shutil.copy(fn, os.path.join("latency_debug", "all", new_name))
    tw = 2


if __name__ == "__main__":
    main()

