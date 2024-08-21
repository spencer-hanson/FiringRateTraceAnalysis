import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import get_xaxis_vals, confidence_interval
from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import calc_quandist
from population_analysis.plotting.distance.fraction_distance_significant import get_session_significant_timepoint_list
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup

DISTANCES_LOCATION = "D:\\PopulationAnalysisDists"


def ensure_rpextra_exists(fn, sess, cache_filename, quan):
    if os.path.exists(fn):
        print(f"Loading precalculated rpextra dist '{fn}'..")
        return True, 1
    try:
        print(f"No precalculated dist exists for '{fn}' calculating..")
        rpperi = sess.rp_peri_units().shape[1]
        rpextra = len(sess.trial_filter_rp_extra().idxs())
        prop = rpperi / rpextra
        prop = prop / 10  # divide by 10 since we have 10 latencies
        prop = prop / 2  # divide by 2 since we have 2 directions  TODO find proportion of directions

        motions = [1]
        calc_quandist(sess, sess.unit_filter_premade(), sess.trial_filter_rp_extra(), cache_filename, quan=quan, use_cached=True, base_prop=prop, motions=motions)
        return True, 1
    except Exception as e:
        print(f"Error ensuring rpextra dist exists! Error: '{str(e)}'")
        return False, e


def frac_sig_dist_euc_max_vals_bars(sess_group, confidence_val):
    olddir = os.getcwd()
    os.chdir(DISTANCES_LOCATION)
    quan = EuclidianQuantification()

    num_sessions = 0
    motdir = 1
    latencies = {}  # {'-500,-400': <count of sessions outside of 99th percentile conf interval>, '-400,-300': ...}

    def add_to_dict(d, k):
        if k not in d:
            d[k] = 0
        d[k] = d[k] + 1

    for filename, sess in sess_group.session_iter():
        if sess is None:
            tw = 2
            continue
        rpextra_error_distribution_fn = f"{filename}-{quan.get_name()}{motdir}.pickle"
        res = ensure_rpextra_exists(rpextra_error_distribution_fn, sess, filename, quan)
        if not res[0]:
            print(f"Error calculating RpExtra distance distribution for '{filename}'.. Skipping..")
            raise res[1]
            time.sleep(2)
            continue
        num_sessions = num_sessions + 1
        print(f"Processing session '{filename}'")

        with open(rpextra_error_distribution_fn, "rb") as f:
            rpextra_error_distribution = pickle.load(f)

        mixed_rel_timestamps = None
        rpperi = None
        rpextra = None
        skip_latency = False
        mmax = 10
        for i in range(mmax):
            if skip_latency:
                continue
            st = (i - (mmax/2)) / 10
            end = ((i - (mmax/2)) / 10) + .1
            rnd = lambda x: int(x*1000)
            latency_key = f"{rnd(st)},{rnd(end)}"
            latency_dist_fn = f"{latency_key}-dists-{quan.get_name()}-{filename}-dir{motdir}.pickle"
            if os.path.exists(latency_dist_fn):
                print(f"Precalculated latency {latency_key} found..")
                with open(latency_dist_fn, "rb") as f:
                    distances = pickle.load(f)
            else:

                if mixed_rel_timestamps is None:  # For faster plotting when data is cached
                    # code date 8/13/24 mixed_rel_timestamps is saccade - probe, so we're going to invert for probe - saccade
                    mixed_rel_timestamps = sess.nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]
                    mixed_rel_timestamps = mixed_rel_timestamps * -1

                    print("Processing unit filter..")
                    ufilt = sess.unit_filter_premade()
                    if len(ufilt.idxs()) == 0 or len(sess.trial_filter_rp_extra().idxs()) == 0:
                        num_sessions = num_sessions - 1
                        print(f"No passing units found in session '{filename}' Skipping..")
                        skip_latency = True
                        continue

                    rpperi = sess.rp_peri_units()[ufilt.idxs()]
                    rpextra = sess.units()[ufilt.idxs()][:, sess.trial_filter_rp_extra().idxs()]

                print(f"Calculating latency {latency_key} distances..", end="")
                lt = mixed_rel_timestamps >= st
                gt = mixed_rel_timestamps <= end
                andd = np.logical_and(lt, gt)
                if not np.any(andd):
                    print(f"No latency data found for {latency_key} for session '{filename}, skipping..'")
                    continue

                distances = []
                for t in range(NUM_FIRINGRATE_SAMPLES):
                    rpp = rpperi[:, andd, t]
                    rpe = rpextra[:, :, t]
                    distances.append(quan.calculate(rpp, rpe))
                print("saving..", end="")
                distances = np.array(distances)
                with open(latency_dist_fn, "wb") as f:
                    pickle.dump(distances, f)
                print("done")
            start, stop = 8, 12
            max_dist, timpt = sorted(list(zip(np.array(distances)[start:stop], range(start, stop))), key=lambda x: x[0])[-1]  # Find the maximum distance between timepoints 8-18

            lower, upper = confidence_interval(rpextra_error_distribution[:, timpt], confidence_val)
            if latency_key not in latencies:
                latencies[latency_key] = 0

            if max_dist > upper:
                add_to_dict(latencies, latency_key)

    # ax.title.set_text(f"Fraction of sessions outside of the {int(confidence_val*100)}th percentile baseline")
    # # ax.title.set_text(f"{quan.get_name()} % sessions with distance above a {confidence_val} interval motion {motdir}")
    # ax.plot(get_xaxis_vals(), session_counts/num_sessions)
    # ax.set_ylabel("% of total sessions")
    # ax.set_xlabel("Time (ms)")
    os.chdir(olddir)
    # plt.savefig("euclidian-significance.svg")
    # plt.show()

    fix, ax = plt.subplots()
    vals = sorted(list(latencies.items()), key=lambda x: int(x[0].split(",")[0]))
    vals = np.array(vals)
    yvals = []
    for idx in range(vals.shape[0]):
        yvals.append(int(vals[idx, 1]) / num_sessions)
    ax.bar(vals[:, 0], yvals)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=.2)
    plt.title(f"Fraction of significant sessions >{confidence_val}")
    plt.show()
    with open("fraction-significant-distances-latency.pickle", "wb") as f:
        pickle.dump(yvals, f)
    tw = 2


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    # grp = NWBSessionGroup("E:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    # grp = NWBSessionGroup("../../../../scripts/mlati10*07-06*")
    grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs")
    # grp = NWBSessionGroup("C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\scripts\\nwbs")

    confidence_val = 0.50
    frac_sig_dist_euc_max_vals_bars(grp, confidence_val)


if __name__ == "__main__":
    main()
