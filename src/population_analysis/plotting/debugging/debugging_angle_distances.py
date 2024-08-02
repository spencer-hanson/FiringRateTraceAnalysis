import math
import os
import glob

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from population_analysis.processors.filters import BasicFilter
from population_analysis.quantification.angle import AngleQuantification
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_units(unitdata1, unitdata2, savename):
    fig, axs = plt.subplots(nrows=2)
    for uidx in range(unitdata1.shape[0]):
        u1mean = np.mean(unitdata1[uidx], axis=0)
        axs[0].plot(u1mean)

        u2mean = np.mean(unitdata2[uidx], axis=0)
        axs[1].plot(u2mean)
    plt.title("all units")
    plt.savefig("all-units-" + savename)
    plt.show()


def plot_means(unitdata1, unitdata2, savename):
    fig, ax = plt.subplots()
    ax.plot(np.mean(np.mean(unitdata1, axis=0), axis=0))
    ax.plot(np.mean(np.mean(unitdata2, axis=0), axis=0))
    plt.title("unit means")
    plt.savefig("unit-means-" + savename)
    plt.show()


def plot_angledist(unitdata1, unitdata2, savename):
    ang = AngleQuantification()
    dists = []
    for t in range(35):
        dists.append(ang.calculate(unitdata1[:, :, t], unitdata2[:, :, t]))

    fig, ax = plt.subplots()
    plt.title("angle dist")
    ax.plot(dists)
    plt.savefig("angledist-" + savename)
    plt.show()


def plot_timepoint(unitdata1, unitdata2, t, savename):
    mean1 = np.mean(unitdata1, axis=1)[:, t]
    mean2 = np.mean(unitdata2, axis=1)[:, t]
    fig, ax = plt.subplots()

    plt.title(f"timepoint {t}")
    ax.plot(mean1)
    ax.plot(mean2)
    plt.savefig(f"timepoint-{t}-{savename}")
    plt.show()


def plot_dist_debug(unitdata1, unitdata2, savename):
    vals_to_track = {
        "mag1": [],
        "mag2": [],
        "dot": [],
        "mag1*mag2": [],
        "dot/(mag1*mag2)": [],
        "math.acos(dot/(mag1*mag2))": []
    }

    fig, axs = plt.subplots(nrows=len(vals_to_track.keys())+1, figsize=(4, 10))
    for t in range(35):

        mean_1 = np.average(unitdata1, axis=1)[:, t]
        mean_2 = np.average(unitdata2, axis=1)[:, t]
        axs[0].plot(mean_1)
        axs[0].plot(mean_2)

        mag1 = np.linalg.norm(mean_1)
        mag2 = np.linalg.norm(mean_2)
        dot = np.dot(mean_1, mean_2)

        for expr in list(vals_to_track.keys()):
            vals_to_track[expr].append(eval(expr))  # NEVER DO THIS (*does it anyways*)

    i = 1
    for name, data in vals_to_track.items():
        axs[i].plot(data)
        axs[i].set_title(name)
        i = i + 1

    plt.savefig("angle-steps-" + savename)
    plt.show()


def plot_dist_with_offset(unitdata1, unitdata2, savename):
    def scale_vec(vec):
        m = min(np.abs(vec))
        if m < 1:
            scale = (1/m)
            result = vec * scale
            return result
        else:
            return vec
        # sig_offs = []
        # for v in vec:
        #     if v < 0:
        #         sig_offs.append(-1)
        #     elif v > 0:
        #         sig_offs.append(1)
        #     else:
        #         sig_offs.append(0)
        # return np.array(sig_offs)


    unsigned_datas = []
    signed_datas = []

    for t in range(35):
        mean_1 = np.average(unitdata1, axis=1)[:, t]
        mean_2 = np.average(unitdata2, axis=1)[:, t]

        umag1 = np.linalg.norm(mean_1)
        umag2 = np.linalg.norm(mean_2)
        udot = np.dot(mean_1, mean_2)
        utheta = math.acos(udot/(umag1*umag2))
        unsigned_datas.append(utheta)

        scaled_m1 = scale_vec(mean_1)
        scaled_m2 = scale_vec(mean_2)

        smag1 = np.linalg.norm(scaled_m1)
        smag2 = np.linalg.norm(scaled_m2)
        sdot = np.dot(scaled_m1, scaled_m2)
        stheta = math.acos(sdot / (smag1 * smag2))
        signed_datas.append(stheta)

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(unsigned_datas)
    axs[1].plot(signed_datas)
    plt.savefig("scaled-" + savename)
    plt.show()


def debug_angle_dists(unitdata1, unitdata2, savename, folder):
    os.chdir(folder)
    plot_units(unitdata1, unitdata2, savename)
    plot_means(unitdata1, unitdata2, savename)
    plot_angledist(unitdata1, unitdata2, savename)
    plot_timepoint(unitdata1, unitdata2, 0, savename)
    plot_dist_debug(unitdata1, unitdata2, savename)
    plot_dist_with_offset(unitdata1, unitdata2, savename)
    os.chdir("../")


def main():
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    # nwbfiles = glob.glob("../../../../scripts/*/*.nwb")
    # nwb_filename = nwbfiles[0]
    # filepath = os.path.dirname(nwb_filename)
    # filename = os.path.basename(nwb_filename)[:-len(".nwb")]
    # sess = NWBSession(filepath, filename)
    # unit_filter = BasicFilter.empty(sess.num_units)
    # unit_filter = sess.unit_filter_premade()
    # unit_filter = BasicFilter([0, 1], 2)
    matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening
    # unitdata1 = sess.units()[unit_filter.idxs()][:, sess.trial_filter_rp_extra().idxs()]
    # unitdata2 = sess.rp_peri_units()[unit_filter.idxs()][:, sess.trial_filter_rp_peri().idxs()]
    def plot_angleplots_with_baseline(mean):
        fakedata = np.random.normal(mean, 2, size=(10, 500, 35))
        fakedata[:, :, 10:20] += 2

        fakedata2 = np.random.normal(mean, 2, size=(10, 500, 35))
        fakedata2[:, :, 10:20] += 4

        unitdata1 = fakedata
        unitdata2 = fakedata2
        if not os.path.exists(str(mean)):
            os.mkdir(str(mean))

        debug_angle_dists(unitdata1, unitdata2, f"mean-{mean}.png", str(mean))

    plot_angleplots_with_baseline(0)
    plot_angleplots_with_baseline(1)
    plot_angleplots_with_baseline(-1)


if __name__ == "__main__":
    main()

