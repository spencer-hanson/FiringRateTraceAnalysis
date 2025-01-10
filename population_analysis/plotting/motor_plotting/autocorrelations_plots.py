from population_analysis.sessions.saccadic_modulation import NWBSession


def autocorrelations_plot(sess: NWBSession):
    # plot by plot by time excluding itself and reverses
    tw = 2


def main():
    visomotor_sessions = ["2023-04-11", "2023-04-12", "2023-04-13", "2023-04-14", "2023-04-17", "2023-04-19",
                          "2023-04-20", "2023-04-21", "2023-04-24", "2023-04-25"]

    sess_fn = visomotor_sessions[0]
    sess = NWBSession(sess_fn)
    autocorrelations_plot(sess)


if __name__ == "__main__":
    main()

