import glob
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from population_analysis.consts import NUM_FIRINGRATE_SAMPLES
from population_analysis.plotting.figures.rpp_rpe_pca_2d import rpp_rpe_pca
from population_analysis.sessions.saccadic_modulation import NWBSession


def rpp_rpe_pca_3d(sess, title):
    pca, rpp_path, rpe_path = rpp_rpe_pca(sess, 3)
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(rpp_path[:, 0], rpp_path[:, 1], rpp_path[:, 2], color="orange", label="RpPeri")  # plt.get_cmap("autumn")
    ax.plot(rpe_path[:, 0], rpe_path[:, 1], rpe_path[:, 2], color="blue", label="RpExtra")  # plt.get_cmap("winter")
    ax.scatter(rpp_path[:, 0], rpp_path[:, 1], rpp_path[:, 2], color="orange")
    ax.scatter(rpe_path[:, 0], rpe_path[:, 1], rpe_path[:, 2], color="blue")
    plt.title(title)
    plt.legend()
    plt.show()

    print("Saving to file")
    with open("neural-trajectories-data.pickle", "wb") as f:
        pickle.dump({
            "rp_extra_path": rpe_path,
            "rp_peri_path": rpp_path
        }, f)


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    # nwbfiles = glob.glob("../../../../scripts/*/*.nwb")
    # nwbfiles = glob.glob("../../../../scripts/*/*07-19*.nwb")
    # nwbfiles = glob.glob("../../../../scripts/*/*generated*.nwb")
    # nwbfiles = glob.glob("C:\\Users\\Matrix\\Downloads\\tmp\\*04-14*.nwb")
    nwbfiles = glob.glob("C:\\Users\\Matrix\\Downloads\\tmp\\*05-15*.nwb")

    for nwb_filename in nwbfiles:
        try:
            filepath = os.path.dirname(nwb_filename)
            filename = os.path.basename(nwb_filename)[:-len(".nwb")]

            sess = NWBSession(filepath, filename)
            rpp_rpe_pca_3d(sess, nwb_filename)
        except Exception as e:
            print(f"Error with file '{nwb_filename}' Skipping.. Error '{str(e)}'")
            continue


if __name__ == "__main__":
    main()
