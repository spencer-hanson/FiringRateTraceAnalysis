import os.path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from population_analysis.consts import METRIC_NAMES, METRIC_THRESHOLDS


def waveform_generator(unit_num, timestamp_offset_in_ms, suppress=False, disable=False):
    # Draw for a reference https://virtual-graph-paper.com/
    waveform_shape = [
        (0, .01),  # (time ms, probability of spiking)
        (210, .15),  # until next time is reached, previous probability will be used
        (230, .25),
        (250, .35),
        (260, .4),
        (280, .35),
        (300, .3),
        (330, .2),
        (350, .1),
        (370, .05),
        (390, .01),
        (400, .01)
    ]
    if suppress:
        new_wv = []
        for wvt, wvp in waveform_shape:
            new_wv.append([wvt, .5*wvp])  # Reduce probability of spiking by 50%
        waveform_shape = new_wv
    if disable:
        new_wv = []
        for wvt, wvp in waveform_shape:
            new_wv.append([wvt, 0.1*wvp])
        waveform_shape = new_wv

    cur_idx = 0
    spike_probs = []  # list of spike probabilities, 700 long
    for t in range(700):
        if cur_idx + 1 > len(waveform_shape) - 1:
            spike_probs.append(waveform_shape[-1][1])  # Add last probability in the list
            continue
        else:
            next_tval, _ = waveform_shape[cur_idx + 1]

            if t >= next_tval:
                cur_idx = cur_idx + 1

            tval, prob = waveform_shape[cur_idx]
            spike_probs.append(prob)
    # plt.plot(spike_probs)
    # plt.show()

    spikes = []
    times = []
    for t in range(700):
        val = np.random.uniform(low=0, high=1)
        if val <= spike_probs[t]:
            spikes.append(unit_num)
            times.append(timestamp_offset_in_ms + t)
    times = np.array(times)/1000  # Convert from ms

    return spikes, times


def save_in_h5file(h5file, path, data):
    split = path.split("/")
    name = split[-1]

    val = h5file
    for sp in split[:-1]:
        if sp in val.keys():
            val = val[sp]
        else:
            val.create_group(sp)
            val = val[sp]

    val.create_dataset(name, data=np.array(data))


def generate_session(num_units):
    # 300 trials for 100 Rs, 100 RpExtra, 100 Rmixed
    # 700ms * 300 trials(motdir -1,1) +  300trials*10ms for blank screen
    # |--trial1--|-10ms-|--trial2--| ... -|-10ms-|
    print("Starting")

    spike_clusters = []
    spike_times = []
    motion_directions = []
    probe_times = []
    saccade_times = []
    dg_times = []
    iti_times = []

    spike_clusters.append(0)
    spike_times.append(0)  # Add a spike time at beginning
    base_offset = 60000  # 60 seconds before spiking starts
    last_offset = -1

    for i in range(300):
        offset = i * 710 + base_offset  # 700 + 10ms
        last_offset = offset
        suppress = False
        disable = False

        motion_directions.append(-1 if i % 2 == 0 else 1)
        dg_times.append(offset/1000)
        iti_times.append((offset+700)/1000)

        mod3 = i % 3
        if mod3 == 0 or mod3 == 2:  # probe or mixed
            probe_times.append((offset + 200)/1000)  # probe starts at 200

        if mod3 == 1 or mod3 == 0:
            vv = (offset + 200)/1000
            saccade_times.append([vv, vv + .01])  # Saccade timestamps is range of start, stop TODO?
            suppress = True

        if mod3 == 0:
            disable = True

        for unit_num in range(num_units):
            sp, tm = waveform_generator(unit_num, offset, suppress=suppress, disable=disable)
            spike_clusters.extend(sp)
            spike_times.extend(tm)
    spike_clusters.append(0)
    spike_times.append(last_offset/1000 + 20)  # Add a spike time all the way to the end to ensure recording is full

    if os.path.exists("generated.hdf"):
        print("Exists, removing..")
        os.remove("generated.hdf")

    h5file = h5py.File("generated.hdf", "w")
    save_in_h5file(h5file, "spikes/clusters", spike_clusters)
    save_in_h5file(h5file, "spikes/timestamps", spike_times)
    save_in_h5file(h5file, "zeta/probe/left/p", np.zeros((num_units,)))
    save_in_h5file(h5file, "zeta/saccade/nasal/p", np.zeros((num_units,)))

    for metric, mname in METRIC_NAMES.items():
        if metric == "ql":
            continue
        v1 = METRIC_THRESHOLDS[mname](100)
        if v1:
            metric_vals = np.full((num_units,), 100)
        else:
            metric_vals = np.full((num_units,), -100)

        save_in_h5file(h5file, f"metrics/{metric}", metric_vals)

    save_in_h5file(h5file, "metrics/ql", np.ones(num_units,))
    save_in_h5file(h5file, "stimuli/dg/grating/timestamps", dg_times)
    save_in_h5file(h5file, "stimuli/dg/iti/timestamps", iti_times)
    save_in_h5file(h5file, "stimuli/dg/grating/motion", motion_directions)

    save_in_h5file(h5file, "stimuli/dg/probe/timestamps", probe_times)
    save_in_h5file(h5file, "saccades/predicted/left/timestamps", saccade_times)
    save_in_h5file(h5file, "saccades/predicted/left/labels", np.ones((len(saccade_times),)))
    print("Writing to file..")
    h5file.close()

    print("Done!")
    tw = 2


def main():
    generate_session(10)


if __name__ == "__main__":
    main()
