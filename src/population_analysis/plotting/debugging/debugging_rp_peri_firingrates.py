from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup
import matplotlib.pyplot as plt
import numpy as np


def all_firingrates(sess: NWBSession):
    rpp_names = {}
    rpp_latencies = {}

    mmax = 10
    motdir = 1
    ufilt = sess.unit_filter_premade()

    for i in range(mmax):
        st = (i - (mmax / 2)) / 10
        end = ((i - (mmax / 2)) / 10) + .1
        rnd = lambda x: int(x * 1000)

        latency_key = f"{rnd(st)},{rnd(end)}"
        rpp_names[i] = latency_key
        rpp_latencies[i] = (st, end)

    fig, axs = plt.subplots(ncols=mmax, figsize=(32, 4), sharey=True)

    base_rpp = sess.rp_peri_units()[ufilt.idxs()]
    for i in range(mmax):
        latency = rpp_latencies[i]
        rpp_trial_filt = sess.trial_filter_rp_peri(*latency, sess.trial_motion_filter(motdir))
        rpp = base_rpp[:, rpp_trial_filt.idxs()]
        [axs[i].plot(v) for v in np.mean(rpp, axis=1)]
    plt.show()

    tw = 2


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    # grp = NWBSessionGroup("F:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    grp = NWBSessionGroup("C:\\Users\\Matrix\\Documents\\GitHub\\SaccadePopulationAnalysis\\scripts\\nwbs\\mlati7-2023-05-15-output")
    filename, sess = next(grp.session_iter())
    all_firingrates(sess)


if __name__ == "__main__":
    main()
