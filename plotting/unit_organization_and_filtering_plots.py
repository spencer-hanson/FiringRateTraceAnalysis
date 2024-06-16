# This script is to sort an already generated directory of units into passing or failing folders
# use unit_summary_plot.standard_multi_rasters(sess, UnitFilter.empty(sess.num_units), suppress_passing=True)
import os.path
import shutil
import time

import matplotlib
from graph_judge import GraphJudge

from population_analysis.processors.nwb import NWBSessionProcessor, UnitFilter
from unit_lib_summary_plots import mean_response, mean_response_custom, avg_raster_plot
import numpy as np


def _unit_filename_to_unit_num(source_folderpath) -> dict[int, str]:
    # find and map unit_num -> img filename
    if not os.path.exists(source_folderpath):
        raise ValueError(f"Cannot find source folder {source_folderpath}!")

    mapping = {}
    for file in os.listdir(source_folderpath):
        if file.endswith(".png") and file.startswith("u"):
            parts = file.split("_")
            unit_num = parts[0][1:]  # cut off the 'u' in 'u12_.. .png'  # TODO change me for different things?
            # unit_num = parts[-1][1:-len(".png")]
            mapping[int(unit_num)] = os.path.join(source_folderpath, file)
    return mapping


def _organize(source_folderpath, dest_foldername, unit_filter: UnitFilter, dry_run=False, show_progress=True):
    if os.path.exists(dest_foldername):
        raise ValueError(f"Folder '{dest_foldername}' exists! Remove before running")
    unitfile_mapping = _unit_filename_to_unit_num(source_folderpath)
    passing_path = os.path.join(dest_foldername, "passes")
    failing_path = os.path.join(dest_foldername, "fails")
    if not dry_run:
        os.mkdir(dest_foldername)
        os.mkdir(passing_path)
        os.mkdir(failing_path)

    pass_count = 0
    fail_count = 0

    def pprint(s, end="\n"):
        if show_progress:
            print(s, end=end)

    for unit_num, unit_path in unitfile_mapping.items():
        pprint(f"Checking unit {unit_num} - ", end="")
        unit_filename = os.path.basename(unit_path)

        if unit_filter.passes_abs(unit_num):
            pprint("pass")
            dest = os.path.join(passing_path, unit_filename)
            pass_count = pass_count + 1
        else:
            pprint("fail")
            dest = os.path.join(failing_path, unit_filename)
            fail_count = fail_count + 1
        if not dry_run:
            shutil.copyfile(unit_path, dest)

    print(f"Done, {pass_count} passed, {fail_count} failed")
    if not dry_run:
        time.sleep(1)


def organize_qm(sess, dry_run=False, show_progress=True):
    filt = sess.unit_filter_qm()
    dst = "qm"
    _organize("src", dst, filt, dry_run=dry_run, show_progress=show_progress)
    if not dry_run:
        plot_avgs(sess, filt, dst)


def organize_zeta(sess, dry_run=False, show_progress=True):
    filt = sess.unit_filter_probe_zeta()
    dst = "zeta"
    _organize("src", dst, filt, dry_run=dry_run, show_progress=show_progress)
    if not dry_run:
        plot_avgs(sess, filt, dst)


def organize_activity(sess, spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore, dry_run=False, show_progress=True):
    filt = sess.unit_filter_custom(spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore)
    dst = f"activity_{spike_count_threshold}sp_{trial_threshold}tr_{missing_threshold}ms_{min_missing}mn_{baseline_mean_zscore}_bzm_{baseline_time_std_zscore}bzs"
    _organize("src", dst, filt, dry_run=dry_run, show_progress=show_progress)
    if not dry_run:
        plot_avgs(sess, filt, dst)


def organize_qm_zeta(sess, dry_run=False, show_progress=True):
    passing_unit_filter = sess.unit_filter_qm().append(
        sess.unit_filter_probe_zeta()
    )
    dst = "qm_zeta"
    _organize("src", dst, passing_unit_filter, dry_run=dry_run, show_progress=show_progress)
    if not dry_run:
        plot_avgs(sess, passing_unit_filter, dst)


def organize_qm_zeta_activity(sess, spike_count_threshold, trial_threshold, missing_threshold, min_missing,
                              baseline_mean_zscore, baseline_time_std_zscore, dry_run=False, show_progress=True, skip_avgs=False):
    filt = sess.unit_filter_qm().append(
        sess.unit_filter_probe_zeta()).append(
        sess.unit_filter_custom(spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_mean_zscore, baseline_time_std_zscore)
    )
    dst = f"qm_zeta_activity_{spike_count_threshold}sp_{trial_threshold}tr_{missing_threshold}ms_{min_missing}mn_{baseline_mean_zscore}bzm_{baseline_time_std_zscore}bzs"

    _organize("src", dst, filt, dry_run=dry_run, show_progress=show_progress)
    # _organize("../plotting", dst, filt, dry_run=dry_run, show_progress=show_progress)
    if not dry_run and not skip_avgs:
        plot_avgs(sess, filt, dst)


def plot_avgs(sess, filt: UnitFilter, dst: str):
    # Plot the average responses and rasters for all response types for a given filter
    fileprefix = os.path.join(dst, "passes") + "/"

    print("Plotting mean responses..")
    print("Rmixed..")
    mean_response(sess, "Rmixed - Passing", filt, sess.mixed_trial_idxs, prefix=fileprefix)
    avg_raster_plot(sess, "Rmixed - Passing", filt, sess.mixed_trial_idxs, -1, prefix=fileprefix)

    print("Rs..")
    mean_response(sess, "Rs - Passing", filt, sess.saccade_trial_idxs, prefix=fileprefix)
    avg_raster_plot(sess, "Rs", filt, sess.saccade_trial_idxs, -1, prefix=fileprefix)

    print("Rp_Extra..")
    mean_response(sess, "Rp_Extra - Passing", filt, sess.probe_trial_idxs, prefix=fileprefix)
    avg_raster_plot(sess, "Rp_Extra - Passing", filt, sess.probe_trial_idxs, -1, prefix=fileprefix)

    print("Rp_Peri..")
    mean_response_custom(np.mean(sess.rp_peri_units()[filt.idxs(), :, :], axis=1), "Rp_Peri", prefix=fileprefix)
    print("Done!")


def judge_filters(sess: NWBSessionProcessor):
    gj = GraphJudge.from_directory("src")

    def wrap_filter():
        pass

    pass


def main():
    # shutil.rmtree("qm_zeta")
    filename = "2023-05-15_mlati7_output"
    matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSessionProcessor("../scripts", filename, "../graphs")
    # judge_filters(sess)

    # organize_qm_zeta(sess)
    # dry_run = True
    # show_progress = False
    dry_run = False  # If true will not actually copy files/make directories/graphs
    show_progress = True
    skip_avgs = True  # Skip plotting the averages

    def oo(sp, tr, ms, mn, bzm, bzs):
        print(f"{sp}sp_{tr}tr_{ms}ms_{mn}mn_{bzm}bzm_{bzs}_bzs")
        organize_qm_zeta_activity(
            sess,
            spike_count_threshold=sp,
            trial_threshold=tr,
            missing_threshold=ms,
            min_missing=mn,
            baseline_mean_zscore=bzm,
            baseline_time_std_zscore=bzs,
            # optional
            dry_run=dry_run,
            show_progress=show_progress,
            skip_avgs=skip_avgs
        )
        print("--")

    oo(5, .2, 1, 1, .9, .4)

    # TODO fix when filter has no results??

    # organize_qm(sess, dry_run, show_progress)
    # organize_zeta(sess, dry_run, show_progress)
    # organize_activity(sess, *qm_zeta_activity_params, dry_run=dry_run, show_progress=show_progress)
    # organize_qm_zeta(sess, dry_run, show_progress)

    tw = 2


if __name__ == "__main__":
    main()
