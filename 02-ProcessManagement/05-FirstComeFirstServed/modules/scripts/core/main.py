from .enumerations import SchedulingType
from .implementations import Preliminaries, Simulation, Plotting

import pandas as pd


def main():
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

    from data import LECTURE_EXAMPLES

    procs = pd.DataFrame(LECTURE_EXAMPLES["Finals"])
    procs = procs.set_index("Process")
    procs = procs.drop(columns="Priority No.")

    scheduling = SchedulingType.FCFS

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
    procs_props = sim.processes

    ## Calculate CPU Utilization
    idle_time = procs_props["Idle"].sum()  # Sum of all idle periods
    duration = procs_props["Finish"].max()  # Time by which all processes are finished
    cpu_util = (1 - (idle_time / duration)) * 100

    ## Calculate Average Turnaround Time (ATA)
    ata = procs_props["Turnaround"].mean()

    ## Calculate Average Waiting Time (AWT)
    awt = procs_props["Waiting"].mean()

    ## Events list
    events = sim.events

    ## Event timepoints
    timepoints = sim.timepoints

    # timepoints = procs_props.reset_index().melt(
    #     id_vars=["Process"],
    #     value_vars=["Start", "Finish"],
    #     var_name="Type",
    #     value_name="Time",
    # )
    # ...  # Add idle events
    # timepoints = timepoints.sort_values(by="Time")
    # timepoints = timepoints[["Process", "Type", "Time"]]  # Reorder columns

    ## Gantt chart
    plot = Plotting(procs_props)

    #############
    ## OUTPUTS ##
    #############

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
