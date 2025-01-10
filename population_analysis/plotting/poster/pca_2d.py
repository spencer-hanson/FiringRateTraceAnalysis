import math

from population_analysis.plotting.trajectory.trajectory_pca import train_pca
from population_analysis.processors.filters import BasicFilter, Filter
from population_analysis.processors.filters.unit_filters import UnitFilter
from population_analysis.sessions.saccadic_modulation import NWBSession
import numpy as np
import matplotlib.pyplot as plt


def plot_2d_pca(sess: NWBSession, ufilt: Filter):
    n_components = 2  # 2d graph
    # pca_training_trial_filters is a list of trial filters used to create the sample
    # pca_training_units is a np arr (units, trials, t)

    # (units, t*len(pca_training_trial_filters))
    # pca_training_data = preprocess_data_pca(pca_training_units, pca_training_trial_filters)
    # pca = train_pca(pca_training_data, n_components)

    fig, ax = plt.subplots()

    pcas = []
    datas = [
        (sess.trial_filter_rp_extra().append(sess.trial_motion_filter(1)), sess.units()[ufilt.idxs()], "Oranges", "RpExtra"),
        (sess.trial_filter_rp_peri(-.2, .2, sess.trial_motion_filter(1)), sess.rp_peri_units()[ufilt.idxs()], "Blues", "RpPeri")
    ]

    for response_info in datas:
        # response_info is trial_filter, axis [row,col], name, units [units, trials, t], color
        trial_filter = response_info[0]
        unit_data = response_info[1]

        colorname = response_info[2]
        cmap = plt.get_cmap(colorname)
        # color = lambda x: cmap(1 - float(x) / (35 - 1))
        # color = lambda x: cmap(1/math.pow(math.e, x/2)*35)
        color = lambda x: cmap(math.pow(math.e, x/35) - 1)

        name = response_info[3]

        raw_response_data = unit_data[:, trial_filter.idxs()]

        n_units = raw_response_data.shape[0]
        n_trials = raw_response_data.shape[1]
        n_timepoints = raw_response_data.shape[2]
        n_samples = n_trials * n_timepoints  # Number of all timepoints in all trials

        response_data = raw_response_data.swapaxes(0, 2)  # swap (units, trials, t) to (t, trials, units)
        response_data = response_data.reshape(
            (n_samples, n_units))  # reshape into (x, units) where x is total trials*timepoints
        # Each entry in our list is a timepoint in num_unit-dimensional space
        # TODO test pca mean before train
        pca = train_pca(response_data,
                        n_components)  # train pca and reduce our pop vecs down to 2d, data is (n_samples, num_components)
        pcas.append(pca)
        sample_val = np.mean(raw_response_data, axis=1)  # mean over trials, left with (units, t)
        sample_val = sample_val.swapaxes(0, 1)  # want each timepoint as a point in pca space, being 'units'-dimensional

        r = pca.transform(sample_val)  # returns (t, num_components)

        arr = np.array(r)
        for idx in range(1, arr.shape[0]):
            xval = arr[idx-1:idx+1, 0]
            yval = arr[idx-1:idx+1, 1]
            kwargs = {}
            if idx == 16:
                kwargs = {"label": name}
            ax.plot(xval, yval, color=color(idx-1), linewidth=4, **kwargs)

        tw = 2
    ax.legend()
    ax.set_title("Neural Trajectory")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("PCA 1")
    ax.set_xlabel("PCA 2")
    plt.savefig("pca-2d-trajectory.png", transparent=True)
    print("Saved and done!")


def main():
    fn = "E:\\PopulationAnalysisNWBs\\mlati7-2023-05-15-output\\mlati7-2023-05-15-output.hdf.nwb"
    sess = NWBSession(fn)
    # Small
    ufilt = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.units().shape[1])
    plot_2d_pca(sess, ufilt)


if __name__ == "__main__":
    main()
