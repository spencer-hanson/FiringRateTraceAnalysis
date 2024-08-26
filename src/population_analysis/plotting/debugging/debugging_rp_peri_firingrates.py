from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.group import SessionGroup
import matplotlib.pyplot as plt
import numpy as np


def all_firingrates(sess: NWBSession):
    # dd = {}
    # digitize version (the same as below trial filter as of 8/20/24)
    # offsets = sess.mixed_rel_timestamps
    # offset_bins = np.digitize(offsets, np.arange(-.5, .6, .1))
    # rp_peri = sess.rp_peri_units()[sess.unit_filter_premade().idxs(), :]
    #
    # fig, axs = plt.subplots(ncols=10, figsize=(32, 4), sharex=True, sharey=True)
    #
    # for i in range(10):
    #     offset_idxs = offset_bins == (i + 1)  # plus one since digitize bin num0 is anything less than -.5, likewise num11 is greater than .5 (total digitize range is 0-11, len == 12)
    #     dd[i] = np.where(offset_idxs)[0]
    #
    #     c0 = rp_peri[:, offset_idxs]
    #     c1 = np.mean(c0, axis=1)
    #     c2 = np.mean(c1, axis=0)
    #     axs[i].plot(c2, color="blue", label="raw rpperi")
    #     axs[i].title.set_text(f"{round(np.arange(-.5, .6, .1)[i], 3)}")
    # axs[-1].legend()
    # plt.show()
    # tw = 2

    rpp_names = {}
    rpp_latencies = {}

    mmax = 10
    motdir = 1
    ufilt = sess.unit_filter_premade()
    # ufilt = BasicFilter.empty(sess.num_units)
    # trial filter version (the same above digitize as of 8/20/24)
    for i in range(mmax):
        st = (i - (mmax / 2)) / 10
        end = ((i - (mmax / 2)) / 10) + .1
        rnd = lambda x: int(x * 1000)

        latency_key = f"{rnd(st)},{rnd(end)}"
        rpp_names[i] = latency_key
        rpp_latencies[i] = (st, end)

    fig, axs = plt.subplots(ncols=mmax, figsize=(32, 4), sharey=True, sharex=True)

    base_rpp = sess.rp_peri_units()[ufilt.idxs()]
    for i in range(mmax):
        latency = rpp_latencies[i]
        # rpp_trial_filt = sess.trial_filter_rp_peri(*latency, sess.trial_motion_filter(motdir))
        rpp_trial_filt = sess.trial_filter_rp_peri(*latency)
        # idxs1 = rpp_trial_filt.idxs()
        # idxs2 = dd[i]

        rpp = base_rpp[:, rpp_trial_filt.idxs()]
        # [axs[i].plot(v) for v in np.mean(rpp, axis=1)]  # See a moosh of all units
        axs[i].plot(np.mean(np.mean(rpp, axis=1), axis=0))
    plt.show()


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    # grp = NWBSessionGroup("F:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    grp = SessionGroup("C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\scripts\\nwbs\\tmp")
    filename, sess = next(grp.session_iter())
    all_firingrates(sess)


if __name__ == "__main__":
    main()
