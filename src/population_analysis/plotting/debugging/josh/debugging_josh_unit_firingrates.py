import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.plotting.figures.frac_sig_euc_dist_bars_max_vals import iter_hdfdata, slice_rp_peri_by_latency


def plot_session(sessdict):
    latencies = np.arange(-.5, .6, .1)
    fig, axs = plt.subplots(ncols=len(latencies)-1, figsize=(24, 2))
    xvals = np.arange(0, 700, 10) - 200
    rpe = sessdict["rp_extra"]
    print(f"Plotting '{sessdict['uniquename']}'..")
    for latency_idx in range(len(latencies)-1):
        rpp = sessdict["rp_peri"][:, :, :, latency_idx]
        ax = axs[latency_idx]

        ax.plot(xvals, np.mean(np.mean(rpp, axis=1), axis=0), color="blue", label="RpPeri")
        ax.plot(xvals, np.mean(np.mean(rpe, axis=1), axis=0), color="orange", label="RpExtra")
        ax.set_xlabel("Time from Probe (ms)")
        titletext = f"({round(latencies[latency_idx],2)},{round(latencies[latency_idx+1],2)})"
        if latency_idx == 0:
            ax.set_ylabel(f"Mean firing rate at probe (normalized)")
            titletext = f"{titletext} {sessdict['uniquename']}"

        ax.title.set_text(titletext)

    axs[-1].legend()
    if not os.path.exists("fr_graphs"):
        os.mkdir("fr_graphs")

    plt.savefig(os.path.join("fr_graphs", sessdict["uniquename"] + ".png"))
    # plt.show()
    tw = 2


def main():
    hdf_fn = "E:\\pop_analysis_2024-08-26.hdf"
    nwb_location = "E:\\PopulationAnalysisNWBs"
    sessions = iter_hdfdata(h5py.File(hdf_fn), nwb_location)

    for sess in sessions:
        plot_session(sess)

    # sess = sessions[0]
    # plot_session(sess)


if __name__ == "__main__":
    main()

