import os
import shutil

from population_analysis.processors.filters.unit_filters import UnitFilter
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.plotting.unit.unit_summary import get_foldername


def make_unitmap_from_folder(source_folderpath):
    # find and map unit_num -> img filename
    if not os.path.exists(source_folderpath):
        raise ValueError(f"Cannot find source folder {source_folderpath}!")

    mapping = {}
    for file in os.listdir(source_folderpath):
        if file.endswith(".png") and file.startswith("unit"):
            parts = file.split("-")
            unit_num = parts[-1][:-len(".png")]  # Cut off the png from 'unit-0.png'
            mapping[int(unit_num)] = os.path.join(source_folderpath, file)
    return mapping


def organize_unit_plots(src_folderpath, unit_filter: UnitFilter, clean=False):
    ufilt_name = unit_filter.get_name()
    src_basename = os.path.basename(src_folderpath)
    output_folder = f"{src_basename}-{ufilt_name}"
    if os.path.exists(output_folder):
        if clean:
            shutil.rmtree(output_folder)
        else:
            raise ValueError(f"Folder '{output_folder}' already exists! clear before re-running or set clean=True!")
    unitfile_mapping = make_unitmap_from_folder(src_folderpath)

    passing_path = os.path.join(output_folder, "passes")
    failing_path = os.path.join(output_folder, "fails")

    os.mkdir(output_folder)
    os.mkdir(passing_path)
    os.mkdir(failing_path)

    pass_count = 0
    fail_count = 0

    for unit_num, unit_path in unitfile_mapping.items():
        print(f"Checking unit {unit_num} - ", end="")
        unit_filename = os.path.basename(unit_path)

        if unit_filter.passes_abs(unit_num):
            print("pass")
            dest = os.path.join(passing_path, unit_filename)
            pass_count = pass_count + 1
        else:
            print("fail")
            dest = os.path.join(failing_path, unit_filename)
            fail_count = fail_count + 1

        shutil.copyfile(unit_path, dest)

    print(f"Done, {pass_count} passed, {fail_count} failed")


def main():
    filepath = "../../../../scripts/05-15-2023-output"
    filename = "05-15-2023-output.hdf-nwb"

    # filepath = "../../../../scripts/generated"
    # filename = "generated.hdf-nwb"

    # matplotlib.use('Agg')  # Uncomment to suppress matplotlib window opening
    sess = NWBSession(filepath, filename, "../graphs")

    # For a full description of params see population_analysis.processors.filters.unit_filters.custom#CustomUnitFilter
    spike_count_threshold = 5
    trial_threshold = .2
    missing_threshold = 1
    min_missing = 1
    baseline_mean_zscore = .9
    baseline_time_std_zscore = .4
    clean = True  # If the directory exists, remove all before running

    unit_filter = sess.unit_filter_qm().append(sess.unit_filter_probe_zeta().append(sess.unit_filter_custom(spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore)))

    organize_unit_plots(get_foldername(sess.filename_no_ext), unit_filter, clean=clean)
    tw = 2


if __name__ == "__main__":
    main()
