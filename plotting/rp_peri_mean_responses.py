from population_analysis.processors.nwb import NWBSession


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSession("../scripts", filename, "../graphs")
    unit_filter = sess.unit_filter_qm().append(
        sess.unit_filter_probe_zeta().append(
            sess.unit_filter_custom(5, .2, 1, 1, .9, .4)
        )
    )


if __name__ == "__main__":
    main()
