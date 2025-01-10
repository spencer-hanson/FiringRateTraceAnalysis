import glob
import os
import pickle
import re

import h5py
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy

from population_analysis.quantification import QuanDistribution
from population_analysis.quantification.euclidian import EuclidianQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession

WINDOW_START = 22
WINDOW_END = 36


def get_rpextra_from_nwb(nwb_filename, cluster_ids):
    sess = NWBSession(nwb_filename)
    cluster_map = sess.nwb.processing["behavior"]["unit_labels"].data[:]
    unit_idxs = []
    for clust in cluster_ids:
        unit_idxs.append(np.where(cluster_map == clust)[0][0])
    unit_idxs = np.array(unit_idxs)
    rpextra = sess.units()[unit_idxs][:, sess.trial_filter_rp_extra().idxs()]
    return rpextra


def upsample_rpextra_distrib(rpextra_distrib):
    # rpextra_distrib is going to be (10000, 35) we want (10000, 70)
    new_rpextra = []
    for rpe in rpextra_distrib:
        interp = np.interp(np.arange(0, 70), np.arange(0, 35)*2, rpe)
        new_rpextra.append(interp)
    new_rpextra = np.array(new_rpextra)
    return new_rpextra


def calculate_scaling_factor(josh_rp_extra, spenc_rp_extra):
    # Take in the two arrays and get out (units,) arr for values for each unit
    # both are (units, 1, t) trial avgd resps
    assert josh_rp_extra.shape[0] == spenc_rp_extra.shape[0]
    factors = []
    response_idx_start = 18  # Get the maximum repsonse between these indexes, and use that as the index to compute the scaling factor
    response_idx_end = 35
    response_len = response_idx_end - response_idx_start
    for unum in range(josh_rp_extra.shape[0]):
        josh_resp = np.abs(josh_rp_extra[unum][0][response_idx_start:response_idx_end])
        josh_srt = sorted(zip(range(response_len), josh_resp), key=lambda x: x[1])
        josh_max_idx, josh_max_val = josh_srt[-1]

        spenc_resp = np.abs(spenc_rp_extra[unum][0][response_idx_start:response_idx_end])
        spenc_srt = sorted(zip(range(response_len), spenc_resp), key=lambda x: x[1])
        spenc_max_idx, spenc_max_val = spenc_srt[-1]

        scale_factor = josh_max_val / spenc_max_val
        factors.append(scale_factor)

        # import matplotlib.pyplot as plt
        # plt.plot(josh_rp_extra[unum][0], label="josh", color="orange")
        # plt.plot(spenc_rp_extra[unum][0], label="spencer", color="blue")
        # plt.plot(spenc_rp_extra[unum][0]*scale_factor, label="scaled")
        # plt.legend()
        # plt.show()

    factors = np.array(factors)
    return factors


def get_rpe_quantification_distribution(data_dict, base_prop, cache_filename):
    # Expects rp_extra_units in (units, trials, t)
    quan = EuclidianQuantification()
    if os.path.exists(cache_filename):
        with open(cache_filename, "rb") as f:
            print(f"Loading precalculated RpExtra QuanDistrib '{cache_filename}'..")
            return pickle.load(f)

    if base_prop > 1 or base_prop < 0:
        raise ValueError("Cannot split proportion > 1 or < 0!")
    josh_rp_extra = data_dict["rp_extra"]  # (units, 1, t)
    rp_extra_units = get_rpextra_from_nwb(data_dict["nwb"], data_dict["clusters"])
    upsampled_rpextra = upsample_rpextra_distrib(np.mean(rp_extra_units, axis=1))[:, None, :]
    scaling_factor = calculate_scaling_factor(josh_rp_extra, upsampled_rpextra)

    # Approximate across datasets with a scale factor to get it on the same log scale
    rp_extra_units *= scaling_factor[:, None, None]

    # Debug plot upscaled vs josh's
    # plt.plot(np.mean(upsample_rpextra_distrib(np.mean(rp_extra_units, axis=1)), axis=0))
    # plt.plot(np.mean(np.mean(josh_rp_extra, axis=1), axis=0))
    # plt.show()
    proportion = int(rp_extra_units.shape[1] * base_prop)
    proportion = proportion if proportion > 0 else 1

    print(f"Calculating RpExtra QuanDistrib for '{cache_filename}' with a {proportion}/{rp_extra_units.shape[1]-proportion} split..")
    if base_prop == 1:
        proportion = None  # Use the same with no split
    quan_dist = QuanDistribution(
        rp_extra_units[:, :proportion],
        rp_extra_units[:, proportion:],
        quan
    ).calculate()

    quan_dist = upsample_rpextra_distrib(quan_dist)

    with open(cache_filename, "wb") as f:
        pickle.dump(quan_dist, f)
    return quan_dist  # Will return (10k, t)


def get_latencies():
    return np.arange(-.5, .6, .1)  # size 11


def slice_rp_peri_by_latency(rp_peri, latency_start, latency_end):
    # rp peri is (units, trials, latencies)
    # where latencies are the values at (-.5,-.4),(-.4,-.3),...
    latency_idx = np.where(get_latencies() == latency_start)[0][0]
    return rp_peri[:, :, :, latency_idx]


def get_avg_proportion(rp_peri, rp_extra, latencies):
    # Average the proportion for all latencies
    # prop = 0
    # for idx in range(len(latencies) - 1):
    #     prop = prop + get_proportion(rp_peri, rp_extra, latencies[idx], latencies[idx + 1])
    #
    # prop = prop / (len(latencies) - 1)
    # return prop
    return 3/100
    # return 1


def calc_dists(rp_peri, rp_extra, rpe_null_dist, cache_filename):
    dists = []
    quan = EuclidianQuantification()
    for t in range(rp_peri.shape[-1]):
        dists.append(quan.calculate(rp_peri[:, :, t], rp_extra[:, :, t]))  # Dist is between rp extra vs latencies as t

    dists = np.array(dists)
    rpe_baseline = np.mean(rpe_null_dist[:16], axis=0)  # (70,)
    rpp_baseline = np.mean(dists[:16])  # (70,)
    diff = rpe_baseline - rpp_baseline
    dists = dists + diff

    with open(cache_filename, "wb") as f:
        pickle.dump(dists, f)

    return dists


def get_largest_distance(rp_peri, rp_extra, rpe_null_dist, cache_filename):
    # Args should be (units, trials, t)
    all_dists = calc_dists(rp_peri, rp_extra, np.mean(rpe_null_dist, axis=0), cache_filename)

    zipped = np.array(list(enumerate(all_dists)))
    zipped = zipped[WINDOW_START:WINDOW_END]  # Only consider values from -40ms to +40ms around the probe
    sort = sorted(list(zipped), key=lambda x: x[1])   # Sort by value
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


def calc_p_value(value, distrib):
    # distrib is an arr (10k,)
    # Confidence value code
    # hist = np.histogram(distrib, bins=200)
    #
    # pdf = hist[0] / sum(hist[0])
    # cdf = np.cumsum(pdf)
    # bins = hist[1]
    # bin_idx = np.where(value < bins)[0]
    # if len(bin_idx) == 0:
    #     return 1.0
    # bin_idx = bin_idx[0]
    # pvalue = cdf[bin_idx]

    mean = np.mean(distrib)
    std = np.std(distrib)
    # zscore = value - mean
    # zscore = zscore / std
    pvalue = 1 - scipy.stats.norm(mean, std).cdf(value)
    pvalue = np.clip(pvalue, 0.00001, a_max=None)

    size = WINDOW_END - WINDOW_START
    pvalue = pvalue * size
    return pvalue


def get_latency_passing_counts(data_dict, confidence_interval, null_cache_filename, dist_cache_fn):
    latencies = get_latencies()
    num_latencies = len(latencies)

    rp_extra = data_dict["rp_extra"]  # (units, 1trial, t, latencies)
    rp_peri = data_dict["rp_peri"]  # (units, 1trial, t)

    proportion = get_avg_proportion(rp_peri, rp_extra, latencies)
    rpe_null_dist = get_rpe_quantification_distribution(data_dict, proportion, null_cache_filename)


    # all_dists = calc_dists(rp_peri[:, :, :, 5], rp_extra, np.mean(rpe_null_dist, axis=0))
    # mean = np.mean(rpe_null_dist, axis=0)
    # ss = np.std(rpe_null_dist, axis=0)
    # lower = 2*ss
    # upper = 2*ss
    # plt.plot(all_dists)
    # plt.plot(mean-lower, color="orange", linestyle="dashed")
    # plt.plot(mean+upper, color="orange", linestyle="dashed")
    # plt.plot(mean, color="orange")
    # plt.show()
    # with open("figure-g-data.pickle", "wb") as f:
    #     pickle.dump({
    #         "rpp_rpe_dist": all_dists,
    #         "rpe_null_distrib": rpe_null_dist
    #     }, f)

    counts = []
    pvals = []
    for idx in range(num_latencies-1):
        start = latencies[idx]
        end = latencies[idx + 1]
        latency_str = str(round(start, 2))

        rpp = slice_rp_peri_by_latency(rp_peri, start, end)
        latency_dist_fn = dist_cache_fn[:-len(".pickle")] + latency_str + ".pickle"
        timepoint, dist = get_largest_distance(rpp, rp_extra, np.array(rpe_null_dist), latency_dist_fn)
        lower, mean, upper = calc_confidence_interval(rpe_null_dist, confidence_interval)
        fig, ax = plt.subplots()
        ax.plot(lower, linestyle="dashed", color="orange")
        ax.plot(upper, linestyle="dashed", color="orange")
        ax.plot(mean, color="orange")
        with open(latency_dist_fn, "rb") as f:
            dists = pickle.load(f)
        ax.plot(dists, color="blue")
        ax.title.set_text(latency_str)
        plt.show()

        pval = calc_p_value(dist, rpe_null_dist[:, int(timepoint)])
        pvals.append(pval)

        if dist > upper:
            counts.append(1)
        else:
            counts.append(0)
    counts = np.array(counts)
    return counts, pvals


def get_name_to_nwbfilepath_dict(nwbs_location, unique_dates):
    mapping = {}
    found_nwbs = glob.glob(os.path.join(nwbs_location, "**/*.nwb"), recursive=True)
    for date in unique_dates:
        found = False
        pat = re.compile(f".*{date}.*")
        for nwbpath in found_nwbs:
            nwbfn = os.path.basename(nwbpath)
            match = pat.match(nwbfn)
            if match:
                mapping[date] = nwbpath
                found = True
                break
        if not found:
            raise ValueError(f"Cannot find matching NWB for session date '{date}' in NWB location '{nwbs_location}'")

    return mapping


def iter_hdfdata(hdfdata, nwbs_location):
    all_rpp = np.array(hdfdata["ppths"]["pref"]["real"]["peri"])  # units, time, latencies
    all_rpe = np.array(hdfdata["ppths"]["pref"]["real"]["extra"])  # units, time
    alldates = hdfdata["ukeys"]["date"][:, 0].astype(str)
    allclusters = hdfdata["ukeys"]["cluster"][:, 0].astype(int)
    unique_dates = np.unique(alldates)
    nwb_mapping = get_name_to_nwbfilepath_dict(nwbs_location, unique_dates)

    datas = []
    # unique_dates = ['2023-05-15']  # TODO Remove me
    unique_dates = ['2023-07-11']  # TODO Remove me
    noisy_sessions = [
        "2023-04-11",
        "2023-04-12",
        "2023-04-13",
        "2023-04-14",
        "2023-04-17",
        "2023-04-19",
        "2023-04-20",
        "2023-04-21"
        "2023-05-23",
        "2023-05-29",
        "2023-05-31",
        "2023-07-12",  # Has NaN
        "2023-07-13",
        "2023-07-26",
        "2023-08-01"
    ]
    for name in unique_dates:
        if name in noisy_sessions:
            continue
        unit_idxs = np.where(alldates == name)[0]
        sess_rpp = all_rpp[unit_idxs][:, None, :, :]
        sess_rpe = all_rpe[unit_idxs][:, None, :]
        sess_clusts = allclusters[unit_idxs]

        datas.append({
            "uniquename": name,
            "rp_extra": sess_rpe,  # (units, 1trial, time, 10x100ms latencies) already trial averaged
            "rp_peri": sess_rpp,  # (units, 1, t)
            "nwb": nwb_mapping[name],
            "clusters": sess_clusts
        })

    return datas


def get_passing_fractions(hdfdata, nwbs_location, confidence_interval):
    passing_counts = np.zeros((10,))  # 10 latencies from -.5 to .5 in .1 increments
    pvals = {}
    num_sessions = 0
    for data_dict in iter_hdfdata(hdfdata, nwbs_location):
        num_sessions = num_sessions + 1
        cache_filename = f"rpextra-quandistrib-{data_dict['uniquename']}.pickle"
        dist_cache_filename = f"euclidian-dist-{data_dict['uniquename']}.pickle"

        try:
            session_counts, pval = get_latency_passing_counts(data_dict, confidence_interval, cache_filename, dist_cache_filename)
            pvals[data_dict["uniquename"]] = pval
        except Exception as e:
            print(f"Error with session {data_dict['uniquename']} Error: {str(e)} Skipping..")
            num_sessions -= 1
            raise e
            continue
        passing_counts = passing_counts + session_counts

    passing_fractions = passing_counts / num_sessions
    return passing_fractions, pvals


def plot_fraction_significant(hdfdata, nwbs_location, confidence_interval):
    olddir = os.getcwd()
    if not os.path.exists("frac_sig"):
        os.mkdir("frac_sig")
    os.chdir("frac_sig")

    print("Calculating fraction of sessions..")
    passing_fractions, pvals = get_passing_fractions(hdfdata, nwbs_location, confidence_interval)

    # Geometric mean of all p-values
    fig, ax = plt.subplots()
    pval_arr = np.array(list(pvals.values()))
    geo_mean = [np.log(pval_arr[:, i]).mean() for i in range(pval_arr.shape[1])]
    geo_mean = np.exp(geo_mean)
    pvalue_latency_mean = geo_mean

    # arr_to_plot = np.mean(pval_arr, axis=0)
    plt.bar(get_latencies()[:-1], pvalue_latency_mean, width=0.05)
    plt.yscale("log")
    rnge = np.arange(1, 4, .5)
    plt.yticks(1 / np.array([pow(10, i) for i in rnge]), ["10^-" + str(j) for j in rnge])
    plt.xlabel("Saccade-Probe Latency")
    plt.ylabel("p-value")
    plt.title("p-value of saccade-probe latencies for all sessions")
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.show()
    tw = 2

    # Significant p-vals
    testing_p_value = 0.001
    sigs = []
    for t in range(len(get_latencies()) - 1):
        sig = len(np.where(pval_arr[:, t] < testing_p_value)[0])
        sigs.append(sig)
    sigs = np.array(sigs) / pval_arr.shape[0]
    plt.bar(get_latencies()[:-1], sigs, width=0.05)
    plt.xlabel("Saccade-Probe Latency")
    plt.title(f"Fraction of sessions with pvalue < {testing_p_value}")
    plt.show()

    tw = 2
    # Fraction significant plot
    # fig, ax = plt.subplots()
    # ax.bar(get_latencies()[:-1], passing_fractions, width=0.05)
    # plt.xticks(rotation=90)
    # plt.subplots_adjust(bottom=.2)
    # plt.title(f"Fraction of significant sessions >{confidence_interval}")
    # plt.show()
    # del fig
    # del ax
    # Individual session pvals
    # for name, pval_list in pvals.items():
    #     fig, ax = plt.subplots()
    #     ax.bar(get_latencies()[:-1], pval_list, width=0.05)
    #     ax.title.set_text(f"p-values for latencies in session {name}")
    #     plt.savefig(f"pvals-{name}.png")

    os.chdir(olddir)
    with open("fraction-significant-distances-latency.pickle", "wb") as f1:
        pickle.dump(passing_fractions, f1)
    with open("pvalue-distance-latencies.pickle", "wb") as f2:
        pickle.dump(pvalue_latency_mean, f2)
    with open("fraction-significant-pvalues-latency.pickle", "wb") as f3:
        pickle.dump(sigs, f3)

    tw = 2


def main():
    hdf_fn = "E:\\pop_analysis_2024-08-26.hdf"
    nwbs_location = "E:\\PopulationAnalysisNWBs"
    confidence_interval = 0.95
    # import matplotlib
    # matplotlib.use("Agg")
    plot_fraction_significant(h5py.File(hdf_fn), nwbs_location, confidence_interval)


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
