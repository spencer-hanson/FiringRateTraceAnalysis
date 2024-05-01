# This script is to sort an already generated directory of units into passing or failing folders
# use unit_summary_plot.standard_multi_rasters(sess, UnitFilter.empty(sess.num_units), suppress_passing=True)
import os.path
import shutil
import time

from population_analysis.processors.nwb import NWBSessionProcessor, UnitFilter


def _unit_filename_to_unit_num(source_folderpath) -> dict[int, str]:
    # find and map unit_num -> img filename
    if not os.path.exists(source_folderpath):
        raise ValueError(f"Cannot find source folder {source_folderpath}!")

    mapping = {}
    for file in os.listdir(source_folderpath):
        if file.endswith(".png") and file.startswith("u"):
            parts = file.split("_")
            unit_num = parts[0][1:]  # cut off the 'u' in 'u12_.. .png'
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


def organize_qm_zeta(sess, dry_run=False, show_progress=True):
    passing_unit_filter = sess.qm_unit_filter().append(
        sess.probe_zeta_unit_filter()
    )
    _organize("src", "qm_zeta", passing_unit_filter, dry_run=dry_run, show_progress=show_progress)


def organize_qm_zeta_activity(sess, spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_zscore, dry_run=False, show_progress=True):
    filt = sess.qm_unit_filter().append(sess.probe_zeta_unit_filter()).append(
        sess.activity_threshold_unit_filter(spike_count_threshold, trial_threshold, missing_threshold, min_missing, baseline_zscore)
    )
    dst = f"qm_zeta_activity_{spike_count_threshold}sp_{trial_threshold}tr_{missing_threshold}ms_{min_missing}mn_{baseline_zscore}bz"

    _organize("src", dst, filt, dry_run=dry_run, show_progress=show_progress)
    from unit_summary_plot import mean_response, mean_response_custom, avg_raster_plot
    import numpy as np

    fileprefix = os.path.join(dst, "passes") + "/"
    if not dry_run:
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


def main():
    # shutil.rmtree("qm_zeta")
    filename = "2023-05-15_mlati7_updated_output"
    # matplotlib.use('Agg')   # Uncomment to suppress matplotlib window opening

    sess = NWBSessionProcessor("../scripts", filename, "../graphs")
    # organize_qm_zeta(sess)
    # dry_run = True
    # show_progress = False
    dry_run = False
    show_progress = True

    def oo(sp, tr, ms, mn, bz):
        print(f"{sp}sp_{tr}tr_{ms}ms_{mn}mn_{bz}bz")
        organize_qm_zeta_activity(
            sess, spike_count_threshold=sp,
            trial_threshold=tr,
            missing_threshold=ms,
            min_missing=mn,
            baseline_zscore=bz,
            # optional
            dry_run=dry_run,
            show_progress=show_progress
        )
        print("--")

    # oo(25, .2, 1, 1)
    # oo(8, .1, 1, 1)
    # oo(8, .2, 1, 1)
    oo(8, .5, 1, 1, 2)

    # organize_qm_zeta_activity(sess, spike_count_threshold=20, trial_threshold=.2, missing_threshold=.8, min_missing=5)
    # organize_qm_zeta_activity(sess, spike_count_threshold=15, trial_threshold=.3, missing_threshold=.7, min_missing=5)
    # organize_qm_zeta_activity(sess, spike_count_threshold=10, trial_threshold=.4, missing_threshold=.6, min_missing=5)

    tw = 2


if __name__ == "__main__":
    main()
