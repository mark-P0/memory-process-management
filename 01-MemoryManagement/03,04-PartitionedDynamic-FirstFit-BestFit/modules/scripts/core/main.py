import pandas as pd
import os

from .enumerations import DynamicType, EventType
from .implementations import (
    Preliminaries,
    EventQueue,
    SummaryTable,
    TimePoints,
)


class PartitionedDynamicMEMMGMT(
    Preliminaries,
    EventQueue,
    SummaryTable,
    TimePoints,
):
    jobs: pd.DataFrame
    base_arrival: pd.Timestamp

    ## TODO: Divide this? Infer from single partition table?
    partition_table: pd.DataFrame
    pat: pd.DataFrame
    fat: pd.DataFrame
    total_allocatable_memory: int

    events: pd.DataFrame
    summary_table: pd.DataFrame
    timepoints: list[list]

    dynamic_type: DynamicType
    with_compaction: bool

    def __init__(
        self,
        jobs,
        total_memory,
        os_size,
        dynamic_type,
        with_compaction,
    ):

        ## Create attributes as safe copies
        self.jobs = jobs.copy()

        ## Verify // Normalize // Sanitize inputs
        self._verify_inputs(jobs, total_memory, os_size)
        self.base_arrival, self.jobs = self._normalize_time(self.jobs)
        self.jobs = self.jobs.set_index("Job No.")
        self.jobs = self.jobs.sort_values(by="Arrival Time")

        ## Initialize partition table
        self.partition_table = self._initialize_partition_table(total_memory, os_size)
        self.total_allocatable_memory = total_memory - os_size

        # print(self.partition_table)
        # print(self.pat)
        # print(self.fat)

        ## Set type of dynamic memory management
        self.dynamic_type = dynamic_type

        ## Store "Compaction" flag
        self.with_compaction = with_compaction
        # self.with_compaction = True

        ## Create events list
        ## TODO: Assignment redundant. Remove?
        self.events = self._generate_events(self.jobs, memory=self.partition_table)


def main():
    ############
    ## INPUTS ##
    ############

    # fmt: off
    
    ## Job table
    job_table = pd.DataFrame({
        
        # ## Sample 1.1, First fit
        # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
        # "Size":             [       200,        250,        300],
        # "Arrival Time":     [ "9:00 AM",  "9:10 AM",  "9:30 AM"],
        # "Run Time (min.)":  [        25,         25,         30],

        ## Sample 1.2, Best fit
        ## Best for testing First Fit and Best Fit
        "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
        "Size":             [       200,        250,        100],
        "Arrival Time":     [ "9:00 AM",  "9:10 AM",  "9:30 AM"],
        "Run Time (min.)":  [        25,         25,         30],
        
        # ## Sample 2 (Exam 3; Static/DynamicBestFit)
        # "Job No.":          [   "Job 1",    "Job 2",    "Job 3",    "Job 4",    "Job 5"],
        # "Size":             [        50,        510,        500,        200,         10],
        # "Arrival Time":     [ "9:00 AM", "10:00 AM", "10:30 AM", "12:00 PM",  "1:00 PM"],
        # "Run Time (min.)":  [        75,        180,        150,        100,         10],
        
    })

    ## Sizes
    os_size, memory = (

        ## Sample 1
        32, 640

        # ## Sample 2 (Exam 3; Static | Dynamic, Best Fit, Non-compaction)
        # 312, 1212

    )

    ## Decide whether dynamic partition allocation props
    dynamic_choice = (
        DynamicType.FirstFit
        # DynamicType.BestFit
    )

    enable_compaction = (
        # False
        True
    )

    # fmt: on

    ###############
    ## PROCESSES ##
    ###############

    ## Initialize core process
    dynamic_mem_mgmt = PartitionedDynamicMEMMGMT(
        jobs=job_table,
        total_memory=memory,
        os_size=os_size,
        dynamic_type=dynamic_choice,
        with_compaction=enable_compaction,
    )

    ## Access event list
    events = dynamic_mem_mgmt.events.copy()

    ## Create summary table
    summary_table = dynamic_mem_mgmt.generate_summary_table(
        # include_arrivals=True,
        # normalize=True,
    )

    ## Build timepoint details: Memory map, PAT, FAT, waiting states
    timepoints, base_filename = dynamic_mem_mgmt.generate_timepoint_props()

    #############
    ## OUTPUTS ##
    #############

    ## For colored outputs
    # fmt: off
    os.system("color")
    Deallocation, Compaction, Arrival, Allocation = EventType
    bcolors = {
        "HEADER":       "\033[95m",
        Compaction:     "\033[94m",  # OKBLUE
        "OKCYAN":       "\033[96m",
        Allocation:     "\033[92m",  # OKGREEN
        Arrival:        "\033[93m",  # WARNING
        Deallocation:   "\033[91m",  # FAIL
        "ENDC":         "\033[0m",
        "BOLD":         "\033[1m",
        "UNDERLINE":    "\033[4m",
    }
    # fmt: on

    print("\n", "-" * 32, "\n", sep="")

    print(
        "Given job table:",
        job_table,
        "",
        sep="\n",
    )

    print(
        "Job events:",
        events,
        "",
        sep="\n",
    )

    print(
        "Summary table:",
        summary_table,
        "",
        sep="\n",
    )

    for time, job, event, waiting_state, PAT, FAT, _ in timepoints:
        msg = f"[At {time}, {job} {event} occurs]"
        colored = bcolors[event] + msg + bcolors["ENDC"]
        print(colored)

        print(
            "Waiting jobs:",
            waiting_state,
            "",
            sep="\n",
        )

        print(
            "Partition Allocation Table (PAT):",
            PAT,
            "",
            sep="\n",
        )

        print(
            "Free Allocation Table (PAT):",
            FAT,
            "",
            sep="\n",
        )

    print(
        "Memory maps:",
        f"Available at `{base_filename}`",
        "",
        sep="\n",
    )


if __name__ == "__main__":
    main()
