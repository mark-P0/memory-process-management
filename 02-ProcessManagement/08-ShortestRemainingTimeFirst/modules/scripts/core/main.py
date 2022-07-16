def main():
    from data import LECTURE_EXAMPLES
    from enumerations import ProcessColumns, SchedulingType, EventType
    from implementations import Preliminaries, Simulation, Plotting

    import pandas as pd
    import logging

    ## TODO: Move to config file
    config = {
        "level": logging.DEBUG,
        "format": "[%(levelname)-8s] %(message)s",
    }
    logging.basicConfig(**config)
    # logging.disable(level=logging.CRITICAL)  # Disables(?) logging
    # logging.disable(level=logging.DEBUG)

    logging.info("Running console application...")

    ############
    ## INPUTS ##
    ############

    """
    5 jobs

    Variables:
        Arrival time
        Burst time
        Priority number
    """

    # Process, Burst, Arrival, Priority = ProcessColumns

    key = "Finals"
    procs = pd.DataFrame(LECTURE_EXAMPLES[key])

    ## Set scheduling algorithm
    scheduling = SchedulingType.SRTF

    logging.debug(f"Inputs: {key=} {scheduling.value=}")

    ###############
    ## PROCESSES ##
    ###############

    """
    Produce the following:
        CPU utilization (%)
        Average turnaround time (ATA)
        Average waiting time (AWT)
        Gantt chart

        Timepoints for UI
    """

    ## Pre-check input processes
    prelims = Preliminaries(procs.copy(), scheduling)
    procs_ready = prelims.processes

    ## Simulate processes
    sim = Simulation(procs_ready, scheduling)
    procs_props = sim.processes_full

    ## CPU Utilization
    cpu_util = sim.CPU_util

    ## Calculate Average Turnaround Time (ATA)
    ata = sim.ATA

    ## Calculate Average Waiting Time (AWT)
    awt = sim.AWT

    ## Events list
    events = sim.events

    ## Event timepoints
    timepoints = sim.timepoints

    ## Gantt chart
    plot = Plotting(events)
    plot.save("gantt.png")

    #############
    ## OUTPUTS ##
    #############

    [cpu_util, ata, awt]
    events[events["Type"].isin([EventType.Dispatch, EventType.Resume])]

    # print(
    #     procs,
    #     procs_props,
    #     f"{cpu_util=}",
    #     f"{ata=}",
    #     f"{awt=}",
    #     events,
    #     sep="\n",
    # )
    breakpoint()


if __name__ == "__main__":
    main()
