import matplotlib.pyplot as plt

from population_analysis.plotting.mean_responses.mean_responses_r_mixed import plot_r_mixed_mean_responses
from population_analysis.plotting.mean_responses.mean_responses_rp_extra import plot_rp_extra_mean_responses
from population_analysis.plotting.mean_responses.mean_responses_rp_peri import plot_rp_peri_mean_responses
from population_analysis.plotting.mean_responses.mean_responses_rs import plot_rs_mean_responses
from population_analysis.processors.filters import BasicFilter
from population_analysis.sessions.saccadic_modulation import NWBSession


def plot_multi_mean_responses(sess, unit_filter):
    fig, axs = plt.subplots(2, 4, figsize=(32, 8))
    fig.subplots_adjust(wspace=0.6, hspace=.3)


    datas = [
        ("RpExtra", plot_rp_extra_mean_responses),
        ("Rs", plot_rs_mean_responses),
        ("Rmixed", plot_r_mixed_mean_responses),
        ("RpPeri", plot_rp_peri_mean_responses)
    ]

    idx = 0
    for name, func in datas:
        ax1 = axs[0, idx]
        ax2 = axs[1, idx]

        func(sess, unit_filter, [ax1, ax2])
        if idx == 0:
            ax1.set_ylabel("motion=-1")
            ax2.set_ylabel("motion=1")
        ax2.set_xlabel("Time (20ms bins)")

        ax1.set_title("")
        ax2.set_title("")
        ax1.set_title(f"{name} mean responses")
        idx = idx + 1
    plt.savefig("mean_responses.png")
    plt.show()
    tw = 2


def main():
    # filepath = "../../../../scripts"
    # filename = "new_test"

    filepath = "../../../../scripts/generated"
    filename = "generated.hdf-nwb"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSession(filepath, filename, "../../../../graphs", use_normalized_units=True)
    # sess = NWBSession("../../../../scripts", filename, "../../../../graphs", use_normalized_units=False)
    # unit_filter = sess.unit_filter_qm().append(
    #     sess.unit_filter_probe_zeta().append(
    #         sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
    #     )
    # )

    # unit_filter = BasicFilter([189, 244, 365, 373, 375, 380, 381, 382, 386, 344], sess.num_units)
    unit_filter = BasicFilter.empty(sess.num_units)

    plot_multi_mean_responses(sess, unit_filter)


if __name__ == "__main__":
    main()
