from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup
import matplotlib.pyplot as plt
import numpy as np


def avg_firingrate(sess: NWBSession):
    ufilt = sess.unit_filter_premade()
    rpe = sess.units()[ufilt.idxs()][:, sess.trial_filter_rp_extra().idxs()]
    rpp = sess.rp_peri_units()[ufilt.idxs()]
    fig, ax = plt.subplots()
    ax.plot(np.mean(np.mean(rpe, axis=1), axis=0), color="orange", label="RpExtra")
    ax.plot(np.mean(np.mean(rpp, axis=1), axis=0), color="blue", label="RpPeri")
    ax.legend()
    plt.show()
    tw = 2


def dist_baseline(dist_pkl):
    pass


def main():
    # print("Loading group..")
    # grp = NWBSessionGroup("F:\\PopulationAnalysisNWBs\\mlati10*07-06*")
    # filename, sess = next(grp.session_iter())
    # avg_firingrate(sess)

    fn = ""

if __name__ == "__main__":
    main()
