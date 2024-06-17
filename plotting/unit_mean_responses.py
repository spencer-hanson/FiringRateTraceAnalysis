from population_analysis.processors.nwb import NWBSessionProcessor
from population_analysis.processors.nwb.unit_normalization import UnitNormalizer


def main():
    filename = "2023-05-15_mlati7_output"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSessionProcessor("../scripts", filename, "../graphs")


if __name__ == "__main__":
    main()
