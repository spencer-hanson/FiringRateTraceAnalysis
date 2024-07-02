from population_analysis.sessions.saccadic_modulation import NWBSession


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSession("../../../../scripts", filename, "../graphs")


if __name__ == "__main__":
    main()
