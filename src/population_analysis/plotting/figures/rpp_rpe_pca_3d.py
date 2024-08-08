import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.figures.rpp_rpe_pca_2d import rpp_rpe_pca
from population_analysis.sessions.saccadic_modulation import NWBSession


def rpp_rpe_pca_3d(sess):
    pca, rpp_path, rpe_path = rpp_rpe_pca(sess, 3)
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(rpp_path[:, 0], rpp_path[:, 1], rpp_path[:, 2], color="orange", label="RpPeri")
    ax.plot(rpe_path[:, 0], rpe_path[:, 1], rpe_path[:, 2], color="blue", label="RpExtra")
    ax.scatter(rpp_path[:, 0], rpp_path[:, 1], rpp_path[:, 2], color="orange")
    ax.scatter(rpe_path[:, 0], rpe_path[:, 1], rpe_path[:, 2], color="blue")
    plt.legend()
    plt.show()


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    nwbfiles = glob.glob("../../../../scripts/*/*-04-14*.nwb")
    # nwbfiles = glob.glob("../../../../scripts/*/*generated*.nwb")
    nwb_filename = nwbfiles[0]

    filepath = os.path.dirname(nwb_filename)
    filename = os.path.basename(nwb_filename)[:-len(".nwb")]

    sess = NWBSession(filepath, filename)
    rpp_rpe_pca_3d(sess)


if __name__ == "__main__":
    main()
