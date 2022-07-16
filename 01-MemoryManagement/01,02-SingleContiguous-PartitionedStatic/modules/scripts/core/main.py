import pandas as pd

from .implementations import Preliminaries, EventQueue, SummaryTable, TimePoints


class PartitionedStaticMEMMGMT(
    Preliminaries,  # Input checking, normalizations, construction of PAT
    EventQueue,  # Creation of list of events based on job characteristics
    SummaryTable,  # Summary of the events that will occur
    TimePoints,  # Will create information to be seen at each "timepoint", i.e. event
):
    """
    The "methods" of this class are implemented on the inherited classes above
    """

    jobs: pd.DataFrame
    base_pat: pd.DataFrame
    base_pat_complete: pd.DataFrame

    events: pd.DataFrame
    summary_table: pd.DataFrame
    timepoints: list[list]

    def __init__(
        self,
        jobs: pd.DataFrame,
        total_memory: int,
        os_size: int,
        partitions: dict[int],  # e.g. {'P1': 500, 'P2': 600, 'P3': 700}
    ):
        ## Create attributes as safe copies
        self.jobs = jobs.copy()
        self.total_allocatable_memory = total_memory - os_size

        ## Verify // Normalize // Sanitize inputs
        self._verify_inputs(jobs, total_memory, os_size, partitions)
        self.base_arrival, self.jobs = self._normalize_time(self.jobs)
        self.jobs = self.jobs.set_index("Job No.")

        ## Create static Partition Allocation Table (PAT)
        self.base_pat_complete = self._create_pat(partitions, os_size)
        self.base_pat = self.base_pat_complete.drop(0)  # "True" PAT without OS

        ## Create event list
        self.events = self._generate_events(self.jobs, self.base_pat)


def main():
    ############
    ## INPUTS ##
    ############

    # fmt: off

    ## Job table
    job_table = pd.DataFrame({
        
        # ## Sample 1
        # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
        # "Size":             [         5,         32,         50],
        # "Arrival Time":     [ "9:00 AM",  "9:05 AM",  "9:15 AM"],
        # "Run Time (min.)":  [        10,         20,         25],
        
        ## Sample 2 (Exam)
        "Job No.":          [   "Job 1",    "Job 2",    "Job 3",    "Job 4",    "Job 5"],
        "Size":             [        50,        510,        500,        200,         10],
        "Arrival Time":     [ "9:00 AM", "10:00 AM", "10:30 AM", "12:00 PM",  "1:00 PM"],
        "Run Time (min.)":  [        75,        180,        150,        100,         10],
        
        # ## Jobs that can only fit in a single partition (i.e. Single Contiguous LOL)
        # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
        # "Size":             [       600,        500,        400],
        # "Arrival Time":     [ "9:00 AM",  "9:05 AM",  "9:15 AM"],
        # "Run Time (min.)":  [        10,         20,         25],
        
    })

    ## Partition sizes
    pttn_sizes = {

        # ## Sample 1
        # "P1": 8,
        # "P2": 32,
        # "P3": 32,
        # "P4": 120,
        # "P5": 520,

        ## Sample 2 (Exam)
        "P1": 50,
        "P2": 250,
        "P3": 600,

    }

    ## Memory sizes
    os, memory = (

        # ## Sample 1
        # 312, 1024

        ## Sample 2 (Exam)
        312, 1212
    )

    # fmt: on

    ###############
    ## PROCESSES ##
    ###############

    ## Process instance
    inst = PartitionedStaticMEMMGMT(
        jobs=job_table,
        partitions=pttn_sizes,
        total_memory=memory,
        os_size=os,
    )

    ## Job events
    events = inst.events

    ## Build summary table
    summary_table = inst.generate_summary_table(
        # include_arrivals=True,
    )

    ## Build memory map & PAT for every event timepoint
    timepoints, base_filename = inst.generate_timepoint_props()
    print(timepoints, type(timepoints))

    #############
    ## OUTPUTS ##
    #############

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

    for time, job, event, waiting_state, PAT, *_ in timepoints:
        print(f"[At {time}, {job} {event} occurs]")

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
        "Memory maps:",
        f"Available at `{base_filename}`",
        "",
        sep="\n",
    )


if __name__ == "__main__":
    main()
