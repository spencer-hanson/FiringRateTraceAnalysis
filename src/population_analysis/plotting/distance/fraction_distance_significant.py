import pickle

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import confidence_interval, get_xaxis_vals
from population_analysis.quantification.angle import AngleQuantification
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup
import matplotlib.pyplot as plt
import numpy as np


def get_session_significant_timepoint_list(sess_namedata, quan, motdir, confidence_val):
    folder, filename = sess_namedata
    quan_fmt = f"{filename}-{quan.get_name()}{motdir}.pickle"
    dist_fmt = f"dists-{filename}-{quan.get_name()}{motdir}.pickle"

    try:
        with open(quan_fmt, "rb") as f:
            quandata = pickle.load(f)

        with open(dist_fmt, "rb") as f:
            distdata = pickle.load(f)

        sigs = []
        for t in range(NUM_FIRINGRATE_SAMPLES):
            vals = quandata[:, t]
            vals[np.where(np.isnan(vals))[0]] = 0  # Set nan values to 0 for distance

            _, upper = confidence_interval(vals, confidence_val, plot=False)
            sigs.append(1 if upper < distdata[t] else 0)

        return np.array(sigs)
    except FileNotFoundError:
        print(f"Couldn't find data for session '{filename}' Skipping!")
        return None


def plot_fraction_dist(sess_group, confidence_val):
    quans = [
        EuclidianQuantification(),
        AngleQuantification()
    ]
    session_counts = np.zeros((NUM_FIRINGRATE_SAMPLES,))
    num_sessions = 0
    fig, axs = plt.subplots(nrows=2, ncols=len(quans))

    for row_idx, quan in enumerate(quans):
        for col_idx, motdir in enumerate([-1, 1]):
            for sess_namedata in sess_group.session_names_iter():
                counts = get_session_significant_timepoint_list(sess_namedata, quan, motdir, confidence_val)
                if counts is not None:
                    session_counts = session_counts + counts
                    num_sessions = num_sessions + 1

            ax = axs[row_idx, col_idx]
            ax.title.set_text(f"{quan.get_name()} {motdir}")
            # ax.title.set_text(f"{quan.get_name()} % sessions with distance above a {confidence_val} interval motion {motdir}")
            ax.plot(get_xaxis_vals(), session_counts/num_sessions)
            ax.set_ylabel("% of total sessions")
            ax.set_xlabel("Time (ms)")
    plt.show()
    tw = 2


def main():
    print("Loading group..")
    grp = NWBSessionGroup("../../../../scripts")

    confidence_val = 0.95
    plot_fraction_dist(grp, confidence_val)


if __name__ == "__main__":
    main()

