from population_analysis.processors.experiments.saccadic_modulation.hdf import HDFSessionProcessor


def main():
    raw = HDFSessionProcessor("output.hdf", "mlati7", "session0")
    raw.save_to_nwb("new_test.nwb", load_precalculated=True)
    tw = 2


if __name__ == "__main__":
    main()
