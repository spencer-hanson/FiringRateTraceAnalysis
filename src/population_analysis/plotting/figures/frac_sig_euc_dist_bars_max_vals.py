import glob
import os
import pickle

import h5py
import numpy as np
import matplotlib.pyplot as plt

from population_analysis.quantification import QuanDistribution
from population_analysis.quantification.euclidian import EuclidianQuantification


def get_rpe_quantification_distribution(rp_extra_units, proportion, cache_filename):
    # Expects rp_extra_units in (units, trials, t)
    quan = EuclidianQuantification()
    if os.path.exists(cache_filename):
        with open(cache_filename, "rb") as f:
            print(f"Loading precalculated RpExtra QuanDistrib '{cache_filename}'..")
            return pickle.load(f)
    print(f"Calculating RpExtra QuanDistrib for '{cache_filename}'..")
    quan_dist = QuanDistribution(
        # units,
        # sess.rp_peri_units()[ufilt.idxs()],
        rp_extra_units[:, :proportion],
        rp_extra_units[:, proportion:],
        quan
    )

    with open(cache_filename, "wb") as f:
        pickle.dump(quan_dist, f)
    return quan_dist  # Will return (10k, t)


def get_latencies():
    return np.arange(-.5, .6, .1)  # size 11


def get_rp_peri(hdfdata):
    return np.array([])  # TODO return (units, trials, t)


def get_rp_extra(hdfdata):
    return np.array([])  # TODO


def slice_rp_peri_by_latency(rp_peri, latency_start, latency_end):
    return rp_peri  # TODO


def get_proportion(rp_peri, rp_extra, latency_start, latency_end):
    # should be in (units, trials, t)
    rp_peri = slice_rp_peri_by_latency(rp_peri, latency_start, latency_end)
    rpp = rp_peri.shape[1]
    rpe = rp_extra.shape[1]
    prop = rpp / rpe
    return prop


def get_avg_proportion(rp_peri, rp_extra, latencies):
    # Average the proportion for all latencies
    prop = 0
    for idx in range(len(latencies) - 1):
        prop = prop + get_proportion(rp_peri, rp_extra, latencies[idx], latencies[idx + 1])

    prop = prop / len(latencies)
    return prop


def get_largest_distance(rp_peri, rp_extra):
    # Args should be (units, trials, t)
    quan = EuclidianQuantification()

    all_dists = []
    for t in range(35):
        all_dists.append(quan.calculate(rp_peri[:, :, t], rp_extra[:, :, t]))

    all_dists = np.array(all_dists)[8:12]  # timepoint range to check largest dist in
    zipped = np.array(enumerate(all_dists))
    sort = sorted(zipped, key=lambda x: x[1])   # Sort by value
    largest = sort[-1]

    timepoint = largest[0]
    dist_value = largest[1]

    return timepoint, dist_value


def calc_confidence_interval(data, confidence_val):
    # data is an arr (10k,)
    hist = np.histogram(data, bins=200)

    pdf = hist[0] / sum(hist[0])
    cdf = np.cumsum(pdf)

    lower_idx = np.where(cdf > 1 - confidence_val)[0][0]
    lower = hist[1][lower_idx + 1]

    upper_idx = np.where(cdf > confidence_val)[0][0]
    upper = hist[1][upper_idx + 1]

    mean = np.mean(data, axis=0)
    # plt.plot(hist[1][1:], cdf)
    # plt.vlines(lower, 0, 1.0, color="red")
    # plt.vlines(upper, 0, 1.0, color="red")
    # plt.show()
    return lower, mean, upper


def get_latency_passing_counts(rp_extra, rp_peri, confidence_interval, cache_filename):
    latencies = get_latencies()
    num_latencies = len(latencies)

    proportion = get_avg_proportion(rp_peri, rp_extra, latencies)
    rpe_null_dist = get_rpe_quantification_distribution(rp_extra, proportion, cache_filename)
    counts = []
    for idx in range(num_latencies-1):
        start = latencies[idx]
        end = latencies[idx + 1]
        rpp = slice_rp_peri_by_latency(rp_peri, start, end)
        timepoint, dist = get_largest_distance(rpp, rp_extra)
        lower, mean, upper = calc_confidence_interval(rpe_null_dist, confidence_interval)
        if dist > upper:
            counts.append(1)
        else:
            counts.append(0)
    counts = np.array(counts)

    return counts


def iter_hdfdata(hdfdata):
    # TODO Iterate over the hdfdata obj for each session
    data = {
        "uniquename": "05-15-2023",
        "rp_extra": get_rp_extra(hdfdata),
        "rp_peri": get_rp_peri(hdfdata)
    }
    yield data
    return  # TODO


def get_passing_fractions(hdfdata, confidence_interval):
    passing_counts = np.zeros((10,))  # 10 latencies from -.5 to .5 in .1 increments

    num_sessions = 0
    for data_dict in iter_hdfdata(hdfdata):
        num_sessions = num_sessions + 1
        cache_filename = f"rpextra-quandistrib-{data_dict['uniquename']}.pickle"

        rp_extra = data_dict["rp_extra"]  # (units, trials, t)
        rp_peri = data_dict["rp_peri"]

        session_counts = get_latency_passing_counts(rp_extra, rp_peri, confidence_interval, cache_filename)
        passing_counts = passing_counts + session_counts

    passing_fractions = passing_counts / num_sessions
    return passing_fractions


def plot_fraction_significant(hdfdata, confidence_interval):
    if not os.path.exists("frac_sig"):
        os.mkdir("frac_sig")
    os.chdir("frac_sig")

    pass_fn = "passing_fractions.pickle"
    if os.path.exists(pass_fn):
        with open(pass_fn, "rb") as f:
            print("Found precalculated fractions, loading..")
            passing_fractions = pickle.load(f)
    else:
        print("Have to calculate fraction of sessions..")
        passing_fractions = get_passing_fractions(hdfdata, confidence_interval)

    fix, ax = plt.subplots()
    ax.bar(get_latencies(), passing_fractions)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=.2)
    plt.title(f"Fraction of significant sessions >{confidence_interval}")
    plt.show()

    with open("fraction-significant-distances-latency.pickle", "wb") as f:
        pickle.dump(passing_fractions, f)
    tw = 2


def main():
    hdf_fn = "E:\\myfile.hdf"
    confidence_interval = 0.99
    plot_fraction_significant(h5py.File(hdf_fn), confidence_interval)


if __name__ == "__main__":
    main()


# import os
# import pickle
# import time
#
# import numpy as np
# from matplotlib import pyplot as plt
#
# from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
# from population_analysis.plotting.distance.distance_rpp_rpe_errorbars_plots import confidence_interval
# from population_analysis.plotting.distance.distance_verifiation_by_density_rpe_v_rpe_plots import calc_quandist
# from population_analysis.quantification.euclidian import EuclidianQuantification
# from population_analysis.sessions.group import SessionGroup
#
# DISTANCES_LOCATION = "D:\\PopulationAnalysisDists"
#
#
# def ensure_rpextra_exists(fn, sess, cache_filename, quan):
#     if os.path.exists(fn):
#         print(f"Loading precalculated rpextra dist '{fn}'..")
#         return True, 1
#     try:
#         print(f"No precalculated dist exists for '{fn}' calculating..")
#         rpperi = sess.rp_peri_units().shape[1]
#         rpextra = len(sess.trial_filter_rp_extra().idxs())
#         prop = rpperi / rpextra
#         prop = prop / 10  # divide by 10 since we have 10 latencies
#         prop = prop / 2  # divide by 2 since we have 2 directions  TODO find proportion of directions
#
#         motions = [1]
#         calc_quandist(sess, sess.unit_filter_premade(), sess.trial_filter_rp_extra(), cache_filename, quan=quan, use_cached=True, base_prop=prop, motions=motions)
#         return True, 1
#     except Exception as e:
#         print(f"Error ensuring rpextra dist exists! Error: '{str(e)}'")
#         return False, e
#
#
# def frac_sig_dist_euc_max_vals_bars(sess_group, confidence_val):
#     olddir = os.getcwd()
#     os.chdir(DISTANCES_LOCATION)
#     quan = EuclidianQuantification()
#
#     num_sessions = 0
#     motdir = 1
#     latencies = {}  # {'-500,-400': <count of sessions outside of 99th percentile conf interval>, '-400,-300': ...}
#
#     def add_to_dict(d, k):
#         if k not in d:
#             d[k] = 0
#         d[k] = d[k] + 1
#
#     for filename, sess in sess_group.session_iter():
#         if sess is None:
#             tw = 2
#             continue
#         rpextra_error_distribution_fn = f"{filename}-{quan.get_name()}{motdir}.pickle"
#         res = ensure_rpextra_exists(rpextra_error_distribution_fn, sess, filename, quan)
#         if not res[0]:
#             print(f"Error calculating RpExtra distance distribution for '{filename}'.. Skipping..")
#             raise res[1]
#             time.sleep(2)
#             continue
#         num_sessions = num_sessions + 1
#         print(f"Processing session '{filename}'")
#
#         with open(rpextra_error_distribution_fn, "rb") as f:
#             rpextra_error_distribution = pickle.load(f)
#
#         mixed_rel_timestamps = None
#         rpperi = None
#         rpextra = None
#         skip_latency = False
#         mmax = 10
#         for i in range(mmax):
#             if skip_latency:
#                 continue
#             st = (i - (mmax/2)) / 10
#             end = ((i - (mmax/2)) / 10) + .1
#             rnd = lambda x: int(x*1000)
#             latency_key = f"{rnd(st)},{rnd(end)}"
#             latency_dist_fn = f"{latency_key}-dists-{quan.get_name()}-{filename}-dir{motdir}.pickle"
#             if os.path.exists(latency_dist_fn):
#                 print(f"Precalculated latency {latency_key} found..")
#                 with open(latency_dist_fn, "rb") as f:
#                     distances = pickle.load(f)
#             else:
#
#                 if mixed_rel_timestamps is None:  # For faster plotting when data is cached
#                     # code date 8/13/24 mixed_rel_timestamps is saccade - probe, so we're going to invert for probe - saccade
#                     mixed_rel_timestamps = sess.nwb.processing["behavior"]["mixed-trial-saccade-relative-timestamps"].data[:]
#                     mixed_rel_timestamps = mixed_rel_timestamps * -1
#
#                     print("Processing unit filter..")
#                     ufilt = sess.unit_filter_premade()
#                     if len(ufilt.idxs()) == 0 or len(sess.trial_filter_rp_extra().idxs()) == 0:
#                         num_sessions = num_sessions - 1
#                         print(f"No passing units found in session '{filename}' Skipping..")
#                         skip_latency = True
#                         continue
#
#                     rpperi = sess.rp_peri_units()[ufilt.idxs()]
#                     rpextra = sess.units()[ufilt.idxs()][:, sess.trial_filter_rp_extra().idxs()]
#
#                 print(f"Calculating latency {latency_key} distances..", end="")
#                 lt = mixed_rel_timestamps >= st
#                 gt = mixed_rel_timestamps <= end
#                 andd = np.logical_and(lt, gt)
#                 if not np.any(andd):
#                     print(f"No latency data found for {latency_key} for session '{filename}, skipping..'")
#                     continue
#
#                 distances = []
#                 for t in range(NUM_FIRINGRATE_SAMPLES):
#                     rpp = rpperi[:, andd, t]
#                     rpe = rpextra[:, :, t]
#                     distances.append(quan.calculate(rpp, rpe))
#                 print("saving..", end="")
#                 distances = np.array(distances)
#                 with open(latency_dist_fn, "wb") as f:
#                     pickle.dump(distances, f)
#                 print("done")
#             start, stop = 8, 12
#             max_dist, timpt = sorted(list(zip(np.array(distances)[start:stop], range(start, stop))), key=lambda x: x[0])[-1]  # Find the maximum distance between timepoints 8-18
#
            # lower, upper = confidence_interval(rpextra_error_distribution[:, timpt], confidence_val)
#             if latency_key not in latencies:
#                 latencies[latency_key] = 0
#
#             if max_dist > upper:
#                 add_to_dict(latencies, latency_key)
#
#     # ax.title.set_text(f"Fraction of sessions outside of the {int(confidence_val*100)}th percentile baseline")
#     # # ax.title.set_text(f"{quan.get_name()} % sessions with distance above a {confidence_val} interval motion {motdir}")
#     # ax.plot(get_xaxis_vals(), session_counts/num_sessions)
#     # ax.set_ylabel("% of total sessions")
#     # ax.set_xlabel("Time (ms)")
#     os.chdir(olddir)
#     # plt.savefig("euclidian-significance.svg")
#     # plt.show()
#
#     fix, ax = plt.subplots()
#     vals = sorted(list(latencies.items()), key=lambda x: int(x[0].split(",")[0]))
#     vals = np.array(vals)
#     yvals = []
#     for idx in range(vals.shape[0]):
#         yvals.append(int(vals[idx, 1]) / num_sessions)
#     ax.bar(vals[:, 0], yvals)
#     plt.xticks(rotation=90)
#     plt.subplots_adjust(bottom=.2)
#     plt.title(f"Fraction of significant sessions >{confidence_val}")
#     plt.show()
#     with open("fraction-significant-distances-latency.pickle", "wb") as f:
#         pickle.dump(yvals, f)
#     tw = 2
#
#
# def main():
#     print("Loading group..")
#     # grp = NWBSessionGroup("../../../../scripts")
#     # grp = NWBSessionGroup("E:\\PopulationAnalysisNWBs\\mlati10*07-06*")
#     # grp = NWBSessionGroup("../../../../scripts/mlati10*07-06*")
#     grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs")
#     # from population_analysis.sessions.josh_saccadic_modulation import HDFSession
#     # grp = SessionGroup("D:\\PopulationAnalysisRawHDF", session_cls=HDFSession)
#
#     confidence_val = 0.50
#     frac_sig_dist_euc_max_vals_bars(grp, confidence_val)
#
#
# if __name__ == "__main__":
#     main()
