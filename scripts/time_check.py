import h5py
import matplotlib
import matplotlib.pyplot as plt


def main():
    data = h5py.File("updated_output.hdf")
    probe_timestamps = list(data["stimuli"]["dg"]["probe"]["timestamps"])
    saccade_startstop = list(data["saccades"]["predicted"]["left"]["timestamps"])
    saccade_labels = list(data["saccades"]["predicted"]["left"]["labels"])

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

