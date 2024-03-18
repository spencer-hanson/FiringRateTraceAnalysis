from pynwb import NWBHDF5IO


def main():
    filepath = "../scripts/2023-05-15_mlati7_output.nwb"

    nwbio = NWBHDF5IO(filepath)
    nwb = nwbio.read()

    tw = 2


if __name__ == "__main__":
    main()

