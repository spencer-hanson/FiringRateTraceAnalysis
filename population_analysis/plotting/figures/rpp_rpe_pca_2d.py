import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.sessions.saccadic_modulation import NWBSession


def format_to_trainingdata(units):
    data = []
    for t in range(NUM_FIRINGRATE_SAMPLES):
        for trialidx in range(units.shape[1]):
            data.append(units[:, trialidx, t])
    return np.array(data)


def pca_sess(sess, units1, units2, n_components):
    # units are (units, trials, t) long
    # want to turn each into a vector training example

    print("Preparing training data..", end="")
    # u1data = format_to_trainingdata(units1)
    # u2data = format_to_trainingdata(units2)
    u1data = format_to_trainingdata(np.mean(units1, axis=1)[:, None, :])
    u2data = format_to_trainingdata(np.mean(units2, axis=1)[:, None, :])
    print("done")
    alldata = np.vstack([u1data, u2data])

    pca = PCA(n_components=n_components)
    print("Fitting PCA..", end="")
    pca.fit(alldata)
    print("done")

    u1_path = np.mean(units1, axis=1)  # mean along trials axis shape is now (units, t)
    u2_path = np.mean(units2, axis=1)

    u1pts = pca.transform(u1_path.swapaxes(0, 1))
    u2pts = pca.transform(u2_path.swapaxes(0, 1))

    return pca, np.array(u1pts), np.array(u2pts)


def rpp_rpe_pca(sess, n_components):
    ufilt = sess.unit_filter_premade()
    rpp = sess.rp_peri_units()[ufilt.idxs()][:, sess.trial_filter_rp_peri().idxs()]
    rpe = sess.units()[ufilt.idxs()][:, sess.trial_filter_rp_extra().idxs()]

    pca, rpp_path, rpe_path = pca_sess(sess, rpp, rpe, n_components)
    return pca, rpp_path, rpe_path


def rpp_rpe_pca_2d(sess, title):
    pca, rpp_path, rpe_path = rpp_rpe_pca(sess, 2)
    fig, ax = plt.subplots()
    ax.plot(rpp_path[:, 0], rpp_path[:, 1], color="orange", label="RpPeri")
    ax.plot(rpe_path[:, 0], rpe_path[:, 1], color="blue", label="RpExtra")
    ax.scatter(rpp_path[:, 0], rpp_path[:, 1], color="orange")
    ax.scatter(rpe_path[:, 0], rpe_path[:, 1], color="blue")
    for idx in range(rpe_path.shape[0]):
        ax.text(rpe_path[idx, 0], rpe_path[idx, 1], str(idx))
        ax.text(rpp_path[idx, 0], rpp_path[idx, 1], str(idx))
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    nwbfiles = glob.glob("../../../../scripts/*/*.nwb")
    # nwbfiles = glob.glob("../../../../scripts/*/*generated*.nwb")
    # nwbfiles = glob.glob("C:\\Users\\Matrix\\Downloads\\tmp\\*04-14*.nwb")
    # nwbfiles = glob.glob("C:\\Users\\Matrix\\Downloads\\tmp\\*05-15*.nwb")
    nwb_filename = nwbfiles[0]

    for nwb_filename in nwbfiles:
        try:
            filepath = os.path.dirname(nwb_filename)
            filename = os.path.basename(nwb_filename)[:-len(".nwb")]

            sess = NWBSession(filepath, filename)
            rpp_rpe_pca_2d(sess, nwb_filename)
        except Exception as e:
            print(f"Error with file '{nwb_filename}' Skipping.. Error '{str(e)}'")
            continue


if __name__ == "__main__":
    main()

