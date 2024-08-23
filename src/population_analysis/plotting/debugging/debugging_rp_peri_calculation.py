import pickle

import numpy as np
import matplotlib.pyplot as plt
from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup


def latency_to_index_offset(latency):
    # Take a latency in seconds and turn it into a number of indexes (in 20ms binsize)
    latency = latency * 1000  # convert to ms
    num_bins = latency / 20  # divide by 20ms
    rounded = int(round(num_bins, 1))  # round to whole number and convert to integer
    return rounded


def reset_baseline(units):
    # units is (units, t)
    baseline = np.mean(units[:, 0:10], axis=1)  # baseline is the 200ms before the event
    normalized = units - baseline[:, None]
    return normalized


def recalc_rp_peri(sess: NWBSession):
    print(f"Recalculating RpPeri for session '{sess.filename_no_ext}'..")
    # Get the firing rates in the shape (units, trials, 105)
    # 105 is (-35, [0, 35], 70) where 0 is the probe/saccade so the length is 35+35+35 (in 20ms bins)
    # so we can have the firing rate 700ms before and after our 700ms window
    print("Grabbing firing rates and filtering by trial type..")
    # firing_rates = sess.units()
    firing_rates = sess.nwb.processing["behavior"]["large_range_normalized_firing_rates"].data[:]
    # ufilt = sess.unit_filter_premade()
    # firing_rates = firing_rates[ufilt.idxs()]

    latency_start = -.3
    latency_end = -.2
    # latency_start = .2
    # latency_end = .3

    rs_trfilt = sess.trial_filter_rs().append(sess.trial_motion_filter(1))
    rs = np.copy(firing_rates[:, rs_trfilt.idxs()])
    rs_mean = np.mean(rs, axis=1)

    rmixed_trfilt = sess.trial_filter_rmixed(latency_start, latency_end, sess.trial_motion_filter(1))
    trial_latencies = sess.mixed_rel_timestamps[rmixed_trfilt.idxs()]
    rmixed = firing_rates[:, rmixed_trfilt.idxs()][:, :, 35:35 + 35]

    new_rp_peri = np.copy(rmixed)
    new_rp_peri = np.mean(new_rp_peri, axis=1)  # Average over trials

    # Only grab the response during the duration of the saccade
    sac_start = 35
    sac_end = 35+35
    rs_mean = rs_mean[:, sac_start:sac_end]  # Average over trials
    rs_mean = np.pad(rs_mean, [(0,0),(35, 35)])  # Pad zeros around
    #fig2, axs2 = plt.subplots(ncols=4)
    # axsrs = []
    # for i in range(4):
    #     tfig, tax = plt.subplots()
    #     axsrs.append(tax)
    # axs2 = axsrs
    # [axs2[0].plot(v) for v in rs_mean[:, 35:35+35]]
    # axs2[1].plot(np.mean(rs_mean[:, 35:35+35], axis=0))
    # [axs2[2].plot(v) for v in rs_mean]
    # axs2[3].plot(np.mean(rs_mean, axis=0))
    # plt.show()
    #plt.plot(np.mean(rs_mean[uufilt.idxs()][:, 35:35 + 35], axis=0))
    num_mixed_trials = rmixed.shape[1]

    rs_cumulatives = []
    do_plot = False
    do_plot2 = False

    for unit_num in range(rmixed.shape[0]):  # Iterate over units
        print(f"Calculating unit {unit_num}/{rmixed.shape[0]}..")
        rs_cumulative = np.zeros((35,))  # We add different latency-shifted rs_means to account for each trial
        shifts = []

        for trial_num in range(num_mixed_trials):  # Iterate over rmixed's trials
            trial_latency = trial_latencies[trial_num]
            offset = latency_to_index_offset(trial_latency)  # how far away the probe is from the saccade in 20ms bins (neg is probe before sacc)
            # offset = offset * -1
            # Window starts at 35, then we add offset. Negative means probe before saccade, so a negative offset will
            # make the window move 'left' effectively moving the saccade 'right' and therefore positive to align with the probe
            window_start = 35 + offset  # 35 is the start of the regular window
            window_end = 35 + 35 + offset  # 35 + 35 is the end of the regular window
            shifted_rs = rs_mean[unit_num, window_start:window_end]
            shifts.append(shifted_rs)
            rs_cumulative = rs_cumulative + shifted_rs  # Add our shifted rs_mean to our sum list (not added yet
            if do_plot2:
                fig4, ax4 = plt.subplots(nrows=3) #, figsize=(64, 4))
                ax4[0].plot(rs_mean[unit_num])
                ax4[0].vlines(35+offset, np.min(rs_mean[unit_num]), np.max(rs_mean[unit_num]))
                ax4[0].vlines(35+35+offset, np.min(rs_mean[unit_num]), np.max(rs_mean[unit_num]))
                ax4[1].plot(new_rp_peri[unit_num, :])
                ax4[2].plot(shifted_rs)
                plt.show()
                tw = 2

        # Since we are adding multiple rs_means, we need to average by dividing by the number of trials we just added
        rs_cumulative = rs_cumulative / num_mixed_trials
        rs_cumulatives.append(np.copy(rs_cumulative))

        if do_plot:
            fig3, axs3 = plt.subplots(nrows=2)
            ax3 = axs3[0]
            ax3.plot(new_rp_peri[unit_num, :], color="red")
            ax3.plot(rs_cumulative, color="blue")
            ax3.plot(new_rp_peri[unit_num, :] - rs_cumulative, color="green")
            [axs3[1].plot(s, color="blue") for s in shifts]
            axs3[1].plot(np.mean(shifts, axis=0), color="red")
            plt.show()
            tw = 2

        new_rp_peri[unit_num, :] -= rs_cumulative  # Subtract rs from the mean rmixed

    save_fn = f"newcalc_rp_peri-{sess.filename_no_ext}.pickle"
    print(f"Saving to file '{save_fn}'..")

    with open(save_fn, "wb") as f:
        pickle.dump(new_rp_peri, f)
    new_rp_peri = reset_baseline(new_rp_peri)

    # rpextra = sess.units()[:, sess.trial_filter_rp_extra().idxs()]
    rp_peri2 = np.mean(np.mean(rmixed, axis=1), axis=0) - np.mean(np.array(rs_cumulatives), axis=0)
    ufilt = sess.unit_filter_premade()
    things_to_plot = [
        (np.mean(rmixed, axis=1), "Rmixed"),
        (new_rp_peri, "RppRC"),
        (rs_cumulatives, "Rsc"),
        (np.broadcast_to(rp_peri2[None, :], (new_rp_peri.shape[0], 35)), "RppRC2")
        # (np.mean(rpextra, axis=1), "RpExtra")
    ]

    extra_count = 0
    sidx = len(things_to_plot)+extra_count  # start idx
    fig, axs = plt.subplots(nrows=2, ncols=len(things_to_plot)+extra_count, sharex=True, sharey=False)

    for idx, element in enumerate(things_to_plot):
        data, name = element
        for uidx in ufilt.idxs():
            axs[0][idx].plot(data[uidx])

        axs[1][idx].plot(np.mean(data, axis=0))
        axs[0][idx].title.set_text(name)

    # for unum in range(num_units):
    #     axs[1][sidx].plot(rs_mean[unum])
    plt.show()
    print("Done!")


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    # grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs\\mlati7-2023-05-12-output*")
    # grp = NWBSessionGroup("D:\\tmp")
    grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs\\mlati10-2023-07-25-output*")
    filename, sess = next(grp.session_iter())
    recalc_rp_peri(sess)


if __name__ == "__main__":
    main()


# def recalc_rp_peri(sess: NWBSession):
#     print(f"Recalculating RpPeri for session '{sess.filename_no_ext}'..")
#     # Get the firing rates in the shape (units, trials, 105)
#     # 105 is (-35, [0, 35], 70) where 0 is the probe/saccade so the length is 35+35+35 (in 20ms bins)
#     # so we can have the firing rate 700ms before and after our 700ms window
#     print("Grabbing firing rates and filtering by trial type..")
#     firing_rates = sess.nwb.processing["behavior"]["large_range_normalized_firing_rates"].data[:]
#     rs = firing_rates[:, sess.trial_filter_rs().idxs()]
#     rmixed = firing_rates[:, sess.trial_filter_rmixed().idxs()]
#
#     rs_mean = np.mean(rs, axis=1)  # Average over trials
#
#     trial_latencies = sess.mixed_rel_timestamps  # "mixed relative timestamps" in seconds, relative to the probe (negative means saccade before probe, need to fix TODO)
#     trial_latencies = trial_latencies * -1  # Flip sign so that the trial latencies are relative to the saccade (negative means probe before saccade)
#
#     num_mixed_trials = rmixed.shape[1]
#
#     # Make a copy of Rmixed for rp_peri
#     new_rp_peri = np.copy(firing_rates[:, sess.trial_filter_rmixed().idxs()][:, :, 35:35+35])
#     new_rp_peri = np.mean(new_rp_peri, axis=1)  # Average over trials
#
#     for unit_num in range(rmixed.shape[0]):  # Iterate over units
#         print(f"Calculating unit {unit_num}/{rmixed.shape[0]}..")
#         rs_cumulative = np.zeros((35,))  # We add different latency-shifted rs_means to account for each trial
#         for trial_num in range(num_mixed_trials):  # Iterate over rmixed's trials
#             trial_latency = trial_latencies[trial_num]
#             offset = latency_to_index_offset(trial_latency)  # how far away the probe is from the saccade in 20ms bins (neg is probe before sacc)
#             # offset = offset * -1
#             # Window starts at 35, then we add offset. Negative means probe before saccade, so a negative offset will
#             # make the window move 'left' effectively moving the saccade 'right' and therefore positive to align with the probe
#             window_start = 35 + offset  # 35 is the start of the regular window
#             window_end = 35 + 35 + offset  # 35 + 35 is the end of the regular window
#             shifted_rs = rs_mean[unit_num, window_start:window_end]
#             # new_rp_peri[unit_num, trial_num, :] -= shifted_rs
#             rs_cumulative = rs_cumulative + shifted_rs  # Add our shifted rs_mean to our sum list (not added yet
#         # Since we are adding multiple rs_means, we need to average by dividing by the number of trials we just added
#         rs_cumulative = rs_cumulative / num_mixed_trials
#         new_rp_peri[unit_num, :] -= rs_cumulative  # Subtract rs from the mean rmixed
#
#     save_fn = f"newcalc_rp_peri-{sess.filename_no_ext}.pickle"
#     print(f"Saving to file '{save_fn}'..")
#
#     with open(save_fn, "wb") as f:
#         pickle.dump(new_rp_peri, f)
#
#     print("Done!")
#
#
# def main():
#     print("Loading group..")
#     # grp = NWBSessionGroup("../../../../scripts")
#     grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs\\mlati7-2023-05-12-output*")
#     filename, sess = next(grp.session_iter())
#     recalc_rp_peri(sess)
#
#
# if __name__ == "__main__":
#     main()


# Half-implemented code pls ignore
# for unit_num in range(rmixed.shape[0]):  # Iterate over units
#     for trial_num in range(rmixed.shape[1]):  # Iterate over rmixed's trials
#         trial_latency = trial_latencies[trial_num]
#         latency_in_bins = latency_to_index_offset(trial_latency)  # eg convert 0.015 to 1 (15ms rounded to 20ms bin, is one index offset)
#         latency_in_bins = latency_in_bins * -1  # We need to flip the sign of the latency to offset in the correct direction
#         # since we have (-35, 0, 35) corresponding to [0D, 35A, 35+35B, 35+35+35C]
#         # Where D is -(700+200)ms before the probe (200ms for window offset, 700ms for extra data)
#         # where A = start of the timewindow for the "event" (saccade time or probe time, or if mixed uses probe time)
#         # A is -200ms before the probe, so index 35+(200/20)=35+10=45 is time 0 relative to the "event"
#         # and B is the end of the window, so +500ms after the probe, 35+35= index 70
#         # and C is +(500+700)ms after the probe, (500ms for window offset, 700ms for extra data)
#         # "extra data" is used in case we need to amend the
#         fr = firing_rates[unit_num][:, saccade_trial_idxs][:, :, 35 + latency_in_bins:35 + 35 + latency_in_bins]
#