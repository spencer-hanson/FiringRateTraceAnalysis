import h5py
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.plotting.figures.frac_sig_euc_dist_bars_max_vals import iter_hdfdata, slice_rp_peri_by_latency


def plot_session(sessdict, ax):
    # datas.append({  # TODO Fix naming thing
    #     "uniquename": str(name.astype(str)),
    #     "rp_extra": sess_rpe,  # (units, tr, 10x100ms latencies)
    #     "rp_peri": sess_rpp  # (units, tr, 1)
    # })
    if ax is None:
        fig, ax = plt.subplots()
    latencies = np.arange(-.5, .6, .1)
    rpp = sessdict["rp_peri"]
    rpe = sessdict["rp_extra"]

    ax.plot(latencies[:-1], np.mean(np.mean(rpp, axis=1), axis=0), color="blue", label="RpPeri")
    ax.plot(latencies[:-1], np.broadcast_to(np.mean(np.mean(rpe, axis=1), axis=0), (10,)), color="orange", label="RpExtra")
    ax.set_xlabel("Saccade-Probe Latency")
    ax.set_ylabel("Mean firing rate at probe (normalized)")

    # ax.legend()
    # plt.show()


def main():
    hdf_fn = "E:\\pop_analysis_2024-08-26.hdf"
    sessions = iter_hdfdata(h5py.File(hdf_fn))
    # sess = sessions[0]
    fig, ax = plt.subplots()

    for sess in sessions:
        plot_session(sess, ax)
    plt.show()


if __name__ == "__main__":
    main()

