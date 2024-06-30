import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from population_analysis.processors.filters import BasicFilter
from population_analysis.sessions.saccadic_modulation import NWBSession


def train_pca(data, components):
    pca = PCA(n_components=components)
    pca.fit(data)  # expects (n_samples, n_features)
    # data = pca.transform(data)  # expects and returns (n_samples, n_features)

    return pca


def plot_pca_components(pca, ax, num_to_plot, label_prefix=""):
    for i in range(num_to_plot):
        ax.plot(pca.components_[i], label=f"{label_prefix} PC {i}")
    ax.set_xlabel("Unit num")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def plot_pca_explained_variance(pca, ax, x_offset=0):
    # Variance of the PCA fit as the number of components are included, ie how much of the data is explained by the
    # the first N PCs

    variance = pca.explained_variance_ratio_
    variance = np.cumsum(variance)

    xvals = np.array(range(len(variance)))
    xvals = xvals + x_offset

    ax.bar(xvals, variance)
    ax.set_xlabel("PCA num")
    ax.set_ylabel("% var explained")
    ax.set_xticks(range(len(variance)))


def preprocess_data_pca(pca_training_units, pca_training_trial_filters):
    # pca_training_trial_filters is a list of trial filters used to create the sample
    # pca_training_units is an np arr (units, trials, t)
    processed_units = []

    for trial_filt in pca_training_trial_filters:
        mean_units = np.mean(pca_training_units[:, trial_filt.idxs()], axis=1)
        processed_units.append(mean_units)
    concat_units = np.hstack(processed_units)

    return concat_units


def plot_trajectory_pca(fig, axs, datas, pca_training_units, pca_training_trial_filters):
    n_components = 2  # 2d graph
    # pca_training_trial_filters is a list of trial filters used to create the sample
    # pca_training_units is a np arr (units, trials, t)

    # (units, t*len(pca_training_trial_filters))
    # pca_training_data = preprocess_data_pca(pca_training_units, pca_training_trial_filters)
    # pca = train_pca(pca_training_data, n_components)

    pcas = []

    for response_info in datas:
        # response_info is trial_filter, axis [row,col], name, units [units, trials, t], color
        trial_filter = response_info[0]
        name = response_info[2]
        unit_data = response_info[3]
        color = response_info[4]

        # response_data = preprocess_data_pca(unit_data, [trial_filter])  # (t, units)

        tw = 2

        p_ax = axs[response_info[1][0]][response_info[1][1]]


        raw_response_data = unit_data[:, trial_filter.idxs()]

        n_units = raw_response_data.shape[0]
        n_trials = raw_response_data.shape[1]
        n_timepoints = raw_response_data.shape[2]
        n_samples = n_trials*n_timepoints  # Number of all timepoints in all trials

        response_data = raw_response_data.swapaxes(0, 2)  # swap (units, trials, t) to (t, trials, units)
        response_data = response_data.reshape((n_samples, n_units))  # reshape into (x, units) where x is total trials*timepoints
        # Each entry in our list is a timepoint in num_unit-dimensional space
        # TODO test pca mean before train
        pca = train_pca(response_data, n_components)  # train pca and reduce our pop vecs down to 2d, data is (n_samples, num_components)
        pcas.append(pca)
        sample_val = np.mean(raw_response_data, axis=1)  # mean over trials, left with (units, t)
        sample_val = sample_val.swapaxes(0, 1)  # want each timepoint as a point in pca space, being 'units'-dimensional

        r = pca.transform(sample_val)  # returns (t, num_components)

        arr = np.array(r)
        p_ax.plot(arr[:, 0], arr[:, 1], color=color)
        p_ax.set_title(name)

    return pcas


def plot_trajectory_summary(sess, ufilt):
    fig, axs = plt.subplots(6, 3, figsize=(24, 12))
    fig.subplots_adjust(wspace=.5, hspace=.6)

    datas = [
        (sess.trial_motion_filter(-1).append(sess.trial_filter_rp_extra()), [0, 0], "RpExtra motion=-1", sess.units()[ufilt.idxs()], "blue"),  # trial filter, axis [row,col], name, units [units, trials, t], color
        (sess.trial_motion_filter(1).append(sess.trial_filter_rp_extra()), [1, 0], "RpExtra motion=1", sess.units()[ufilt.idxs()], "blue"),
        (sess.trial_filter_rp_peri(sess.trial_motion_filter(-1)), [0, 1], "RpPeri motion=-1", sess.rp_peri_units()[ufilt.idxs()], "orange"),
        (sess.trial_filter_rp_peri(sess.trial_motion_filter(1)), [1, 1], "RpPeri motion=1", sess.rp_peri_units()[ufilt.idxs()], "orange"),
        # Same plot for both responses
        (sess.trial_filter_rp_peri(sess.trial_motion_filter(-1)), [0, 2], "Rpp-1", sess.rp_peri_units()[ufilt.idxs()], "blue"),
        (sess.trial_motion_filter(-1).append(sess.trial_filter_rp_extra()), [0, 2], "Rpe-1", sess.units()[ufilt.idxs()], "orange"),
        (sess.trial_filter_rp_peri(sess.trial_motion_filter(1)), [1, 2], "Rpp+1", sess.rp_peri_units()[ufilt.idxs()], "blue"),
        (sess.trial_motion_filter(1).append(sess.trial_filter_rp_extra()), [1, 2], "Rpe+1", sess.units()[ufilt.idxs()], "orange"),
    ]

    # Actual trajectories
    pcas = plot_trajectory_pca(fig, axs, datas, sess.units()[ufilt.idxs()], [
        sess.trial_motion_filter(-1).append(sess.trial_filter_rp_extra()),  # 4 groups to split data on and train
        sess.trial_motion_filter(1).append(sess.trial_filter_rp_extra()),
        sess.trial_filter_rp_peri(sess.trial_motion_filter(-1)),
        sess.trial_filter_rp_peri(sess.trial_motion_filter(1))
    ])

    for idx, response_data in enumerate(datas):  # Plot only first 4
        row, col = response_data[1]
        name = response_data[2]

        pca = pcas[idx]

        # Variances explained
        row = row + 2
        ax = axs[row][col]
        plot_pca_explained_variance(pca, ax, x_offset=(idx%2)/5)
        ax.set_title(f"PCA Variance {response_data[2]}")

        # PCs
        row = row + 2  # add two to skip var plots
        ax = axs[row][col]
        plot_pca_components(pca, ax, 2, label_prefix=name)
        ax.set_title(f"PCA Components {response_data[2]}")

    # [a.set_aspect(1.0/a.get_data_ratio(), adjustable="box") for a in axs.ravel()]
    plt.show()


def main():
    filename = "new_test"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession("../../../../scripts", filename, "../graphs")

    ufilt = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.units().shape[1])
    # ufilt = sess.unit_filter_qm().append(
    #     sess.unit_filter_probe_zeta().append(
    #         sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
    #     )
    # )

    plot_trajectory_summary(sess, ufilt)


if __name__ == "__main__":
    main()
