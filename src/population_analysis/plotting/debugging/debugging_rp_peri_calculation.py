import pickle

import numpy as np

from population_analysis.sessions.saccadic_modulation import NWBSession
from population_analysis.sessions.saccadic_modulation.group import NWBSessionGroup


def latency_to_index_offset(latency):
    # Take a latency in seconds and turn it into a number of indexes (in 20ms binsize)
    latency = latency * 1000  # convert to ms
    num_bins = latency / 20  # divide by 20ms
    rounded = int(round(num_bins, 1))  # round to whole number and convert to integer
    return rounded


def recalc_rp_peri(sess: NWBSession):
    print(f"Recalculating RpPeri for session '{sess.filename_no_ext}'..")
    # Get the firing rates in the shape (units, trials, 105)
    # 105 is (-35, [0, 35], 70) where 0 is the probe/saccade so the length is 35+35+35 (in 20ms bins)
    # so we can have the firing rate 700ms before and after our 700ms window
    print("Grabbing firing rates and filtering by trial type..")
    firing_rates = sess.nwb.processing["behavior"]["large_range_normalized_firing_rates"].data[:]
    rs = firing_rates[:, sess.trial_filter_rs().idxs()]
    rmixed = firing_rates[:, sess.trial_filter_rmixed().idxs()]

    rs_mean = np.mean(rs, axis=1)  # Average over trials

    trial_latencies = sess.mixed_rel_timestamps  # "mixed relative timestamps" in seconds, relative to the probe (negative means saccade before probe, need to fix TODO)
    trial_latencies = trial_latencies * -1  # Flip sign so that the trial latencies are relative to the saccade (negative means probe before saccade)

    num_mixed_trials = rmixed.shape[1]

    # Make a copy of Rmixed for rp_peri
    new_rp_peri = np.copy(firing_rates[:, sess.trial_filter_rmixed().idxs()][:, :, 35:35+35])
    new_rp_peri = np.mean(new_rp_peri, axis=1)  # Average over trials

    for unit_num in range(rmixed.shape[0]):  # Iterate over units
        print(f"Calculating unit {unit_num}/{rmixed.shape[0]}..")
        rs_cumulative = np.zeros((35,))  # We add different latency-shifted rs_means to account for each trial
        for trial_num in range(num_mixed_trials):  # Iterate over rmixed's trials
            trial_latency = trial_latencies[trial_num]
            offset = latency_to_index_offset(trial_latency)  # how far away the probe is from the saccade in 20ms bins (neg is probe before sacc)
            # Window starts at 35, then we add offset. Negative means probe before saccade, so a negative offset will
            # make the window move 'left' effectively moving the saccade 'right' and therefore positive to align with the probe
            window_start = 35 + offset  # 35 is the start of the regular window
            window_end = 35 + 35 + offset  # 35 + 35 is the end of the regular window
            shifted_rs = rs_mean[unit_num, window_start:window_end]
            rs_cumulative = rs_cumulative + shifted_rs  # Add our shifted rs_mean to our sum list (not added yet
        # Since we are adding multiple rs_means, we need to average by dividing by the number of trials we just added
        rs_cumulative = rs_cumulative / num_mixed_trials
        new_rp_peri[unit_num, :] -= rs_cumulative  # Subtract rs from the mean rmixed

    save_fn = f"newcalc_rp_peri-{sess.filename_no_ext}.pickle"
    print(f"Saving to file '{save_fn}'..")

    with open(save_fn, "wb") as f:
        pickle.dump(new_rp_peri, f)

    print("Done!")


def main():
    print("Loading group..")
    # grp = NWBSessionGroup("../../../../scripts")
    grp = NWBSessionGroup("D:\\PopulationAnalysisNWBs\\mlati7-2023-05-12-output*")
    filename, sess = next(grp.session_iter())
    recalc_rp_peri(sess)


if __name__ == "__main__":
    main()


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