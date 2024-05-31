from matplotlib import animation

from population_analysis.processors.nwb import NWBSessionProcessor
import matplotlib.pyplot as plt


def ani_unit_trials(sess, trial_idxs, unit_num):
    # Animated trial firing rates for given units
    plots = {"d": []}
    fig, ax = plt.subplots()

    def update(frame):
        print(f"Rendering frame {frame}/{len(trial_idxs)}")
        unit_data = sess.units()[unit_num][trial_idxs][frame]
        to_plot = []
        for pl in plots["d"]:
            pl.remove()

        to_plot.append(ax.plot(unit_data)[0])
        plots["d"] = to_plot
        return plots["d"]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(trial_idxs))
    ani.save(filename=f"u_{unit_num}_trials.gif", writer="pillow")

    tw = 2


def main():
    # shutil.rmtree("qm_zeta")
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening

    sess = NWBSessionProcessor("../scripts", filename, "../graphs")
    ani_unit_trials(sess, sess.probe_trial_idxs, 190)


if __name__ == "__main__":
    main()
