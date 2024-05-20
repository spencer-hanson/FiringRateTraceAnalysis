import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = h5py.File("updated_output.hdf")
    probe_timestamps = np.array(list(data["stimuli"]["dg"]["probe"]["timestamps"]))
    saccade_startstop = np.array(list(data["saccades"]["predicted"]["left"]["timestamps"]))
    saccade_timestamps = np.array([s[0] for s in saccade_startstop])
    saccade_labels = np.array(list(data["saccades"]["predicted"]["left"]["labels"]))

    plt.clf()
    pr_ts = probe_timestamps[np.where(np.invert(np.isnan(probe_timestamps)))[0]]
    sa_ts = saccade_timestamps[np.where(np.invert(np.isnan(saccade_timestamps)))[0]]

    pr = [pr_ts, []]
    sa = [[], sa_ts]
    plt.figure(figsize=(16, 4))
    plt.eventplot(pr, colors="red", linewidths=.05, linelengths=1)
    plt.eventplot(sa, colors="blue", linewidths=.05, linelengths=1)
    plt.show()

    plt.plot(pr_ts, range(len(pr_ts)))
    plt.plot(sa_ts, range(len(sa_ts)))
    plt.show()
    tw = 2

    s_directions = list(data["stimuli"]["dg"]["grating"]["motion"])
    s_timestamps = list(data["stimuli"]["dg"]["grating"]["timestamps"])
    s_iti = list(data["stimuli"]["dg"]["iti"]["timestamps"])
    s_motion = list(data["stimuli"]["dg"]["motion"]["timestamps"])

    # plt.title("Direction and grating flip timestamps")
    # plt.plot(s_timestamps, s_directions, label="timestamps")
    # plt.plot(s_iti, s_directions, label="iti")
    plt.plot(s_motion, s_directions, label="motion")
    # plt.scatter(probe_timestamps, [0 for i in range(len(probe_timestamps))], s=1, label="probe_ts")
    # plt.legend()

    plt.scatter([s[0] for s in saccade_startstop], saccade_labels, color="orange")
    # plt.plot()
    plt.show()
    tw = 2


if __name__ == "__main__":
    main()

"""


bt_orig_ts = [oo[1] for oo in saccade_spike_range_idxs]
pr_ts = [oo[1] for oo in trials["probe"]]
sa_ts = [oo[1] for oo in trials["saccade"]]
mx_ts = [oo["saccade"][1] for oo in trials["mixed"]]
bt_ts = list(sa_ts)
bt_ts.extend(mx_ts)
bt_ts = sorted(bt_ts)
pr = [pr_ts, [], [], [], []]
sa = [[], sa_ts, [], [], []]
mx = [[], [], mx_ts, [], []]
bt = [[], [], [], bt_ts, []]
bt_orig = [[], [], [], [], bt_orig_ts]
plt.figure(figsize=(16, 4))
# probe
plt.eventplot(pr, colors="red", linewidths=.05, linelengths=1)
# saccade demixed
plt.eventplot(sa, colors="blue", linewidths=.05, linelengths=1)
# mixed
plt.eventplot(mx, colors="orange", linewidths=.05, linelengths=1, label="mixed")
# saccade demixed + mixed
plt.eventplot(bt, colors="green", linewidths=0.05, linelengths=1)
# saccade orig
plt.eventplot(bt_orig, colors="purple", linewidths=0.05, linelengths=1)
plt.show()

"""