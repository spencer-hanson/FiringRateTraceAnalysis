from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np


def main():
    filepath = "../scripts/2023-05-15_mlati7_output.nwb"
    # filepath = "../scripts/2023-05-12_mlati7_output.nwb"

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    def plot_data(data_to_plot, title):
        num_units = len(nwb.units)
        time_len = data_to_plot.shape[1]

        units_to_plot = num_units
        # units_to_plot = 30
        for unit_num in range(units_to_plot):
            unit_firingrate = data_to_plot[unit_num]
            plt.plot(range(time_len), unit_firingrate)

        plt.vlines(10, 0, np.max(data_to_plot), color="red", label="Probe time")
        plt.legend()
        plt.title(title)
        plt.show()

    plot_data(np.average(nwb.units["trial_firing_rates"].data[:], axis=1), "All Averaged")

    trial_idxs = [
        ("Saccades", nwb.processing["behavior"]["unit-trial-saccade"].data[:]),
        ("Probes", nwb.processing["behavior"]["unit-trial-probe"].data[:]),
        ("Mixed", nwb.processing["behavior"]["unit-trial-mixed"].data[:])
    ]

    for name, idx_list in trial_idxs:
        plot_data(np.average(nwb.units["trial_firing_rates"].data[:, idx_list], axis=1), name)


    tw = 2


if __name__ == "__main__":
    main()

