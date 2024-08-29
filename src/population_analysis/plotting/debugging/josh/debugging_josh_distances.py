import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.plotting.figures.frac_sig_euc_dist_bars_max_vals import iter_hdfdata, slice_rp_peri_by_latency, \
    calc_confidence_interval
from population_analysis.quantification.euclidian import EuclidianQuantification


def calc_dists(rp_peri, rp_extra):
    dists = []
    quan = EuclidianQuantification()
    for t in range(rp_peri.shape[-1]):
        dists.append(quan.calculate(rp_peri[:, :, t], rp_extra[:, :, t]))  # Dist is between rp extra vs latencies as t

    return dists


def get_rp_extra_dists(name, confidence_interval):
    # cache_filename = os.path.join("frac_sig", f"rpextra-quandistrib-{name}.pickle")
    # cache_filename = "../../figures./" + cache_filename
    # with open(cache_filename, "rb") as f:
    #     dist_distrib = pickle.load(f)
    # lower, mean, upper = calc_confidence_interval(dist_distrib, confidence_interval)
    # return lower, mean, upper
    zeros = np.zeros((70,))  # TODO get null distribution
    return zeros, zeros, zeros


def plot_session(sessdict, confidence_interval):
    # datas.append({
    #     "uniquename": str(name.astype(str)),
    #     "rp_extra": sess_rpe,  # (units, tr, 10x100ms latencies)
    #     "rp_peri": sess_rpp  # (units, tr, 1)
    # })
    xvals = np.arange(0, 700, 10) - 200
    latencies = np.arange(-.5, .6, .1)
    fig, axs = plt.subplots(ncols=len(latencies)-1, figsize=(24, 4))

    for latency_idx in range(len(latencies)-1):
        ax = axs[latency_idx]
        rpp = sessdict["rp_peri"][:, :, :, latency_idx]  # (units, trials, time)
        rpe = sessdict["rp_extra"]  # (units, trials, time)

        rpp_dists = calc_dists(rpp, rpe)
        lower, rpe_dists, upper = get_rp_extra_dists(sessdict["uniquename"], confidence_interval)

        ax.plot(xvals, rpp_dists, label="RpPeri vs RpExtra", color="blue")
        ax.plot(xvals, rpe_dists, label="RpExtra vs RpExtra", color="orange")
        ax.plot(xvals, lower, color="orange", linestyle="dotted")
        ax.plot(xvals, upper, color="orange", linestyle="dotted")
        ax.title.set_text(f"({round(latencies[latency_idx], 2)},{round(latencies[latency_idx + 1], 2)})")
        ax.set_xlabel("Time from probe (ms)")

    axs[0].set_ylabel(f"Euclidian Distance of {sessdict['uniquename']} >{confidence_interval}")
    plt.legend()
    plt.show()
    if not os.path.exists("dist_graphs"):
        os.mkdir("dist_graphs")
    print("Saving to png..")
    plt.savefig(f"dist_graphs/{sessdict['uniquename']}.png")


def main():
    hdf_fn = "E:\\pop_analysis_2024-08-26.hdf"
    nwb_location = "E:\\PopulationAnalysisNWBs"
    sessions = iter_hdfdata(h5py.File(hdf_fn), nwb_location)
    # sess = sessions[0]
    confidence_interval = 0.99
    for sess in sessions:
        plot_session(sess, confidence_interval)


if __name__ == "__main__":
    main()

