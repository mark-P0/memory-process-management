## TODO: Clean imports
import pandas as pd
from .enumerations import PartitionStatus, EventType, STColumn

# import matplotlib
from matplotlib.colors import TABLEAU_COLORS as COLOR_DICT

# from matplotlib.colors import CSS4_COLORS as COLOR_DICT

import io
import os
import random
from string import ascii_letters

from kivymd.app import MDApp


class NoInstances:
    def __init__(self, *args, **kwargs):
        """
        This class, and its direct children,
        are not meant to be instantiated!
        """

        raise NotImplementedError


class Preliminaries(NoInstances):
    ## TODO: Turn into enumerations?
    _job_table_cols = ["Job No.", "Size", "Arrival Time", "Run Time (min.)"]
    _pat_cols = ["Partition No.", "Size", "Location"]

    def _verify_inputs(self, jobs, total_memory, os_size, partitions):
        ## Ensure that all job table columns are as expected
        unexpected = jobs.columns.difference(self._job_table_cols)
        assert (
            not unexpected.any()  # There shouldn't be any unexpected columns
        ), f"Unexpected job table columns: {', '.join(unexpected.to_list())}"

        ## Check continuity of partitions, OS size, and total memory
        pttn_sum = sum(partitions.values())
        inferred_total = os_size + pttn_sum
        assert (
            inferred_total == total_memory
        ), f"Expected memory size {total_memory}, computed {inferred_total} ({pttn_sum=} {os_size=})"

    def _normalize_time(self, jobs):
        ## Convert arrival times to `datetime` objects, for easy manipulation
        arrivals_dt = pd.to_datetime(jobs["Arrival Time"], format="%I:%M %p")

        ## Get earliest arrival time
        base_arrival = arrivals_dt.min()

        ## Subtract "earliest arrival time" from all arrivals
        ## So that it can be designated as "Time 0" (origin, base, etc.)
        ## Divide by unit minute (1 minute) so that it can be expressed in minutes
        jobs["Arrival Time"] = (arrivals_dt - base_arrival) // pd.Timedelta("1m")

        return base_arrival, jobs

    def _normalize_time_undo(self, jobs, columns, base_arrival):
        ## Convert times to `datetime` objects, for easy manipulation
        jobs[columns] = jobs[columns].astype("timedelta64[m]")

        ## "Offset" by base arrival time so that it will be back to its "original" form
        jobs[columns] = jobs[columns] + base_arrival

        ## Apply formating function so that they may become actual strings
        jobs[columns] = jobs[columns].apply(
            lambda time_col: time_col.dt.strftime("%I:%M %p")
        )

        return jobs

    def _create_pat(self, partitions, os_size, os_index=0):
        ## Initialize PAT
        pat = pd.DataFrame(columns=self._pat_cols)

        ## Add partition sizes
        ## OS is effectively "Partition 0"
        size_list = list(partitions.values())
        size_list.insert(os_index, os_size)
        pat["Size"] = size_list

        ## Add starting locations
        endings = pat["Size"].cumsum()  # Cumulative sum are the "ending" locations
        pat["Location"] = endings.shift(
            periods=1, fill_value=0
        )  # Shift "forward" to get "starting" locations; ends of the previous are the starts of the current

        ## Add partition "names", will base on indices except for "OS partition"
        pat["Partition No."] = pat.index.to_series().apply(lambda idx: f"P{idx}")
        pat["Partition No."] = pat["Partition No."].replace({"P0": "OS"})

        ## Initialize partition statuses as available; OS partition is allocated (always)
        pat["Status"] = PartitionStatus.Available
        pat.loc[0, "Status"] = PartitionStatus.Allocated

        return pat


class EventQueue(NoInstances):
    ## TODO: Turn into enumerations?
    _event_cols = ["Job No.", "Time", "Type", "Partition"]

    ## TODO: Refactor?
    ## TODO: Remove `current_job` parameter? It's unused
    def _identify_reallocs(self, executed, current_job=None):
        ## Create safe copy
        executed = executed.copy()

        ## Exclude arrival events
        executed = executed[executed["Type"] != EventType.Arrival]

        ## Sort executed event list by order of precedence (Check enum equivalents)
        ## Add temporary column for actual "value" (precedence) of event types
        executed["_Type_vals"] = executed["Type"].apply(lambda row: row.value)
        executed = executed.sort_values(by=["Partition", "Time", "_Type_vals"])
        executed = executed.drop(columns="_Type_vals")  # Remove temporary column

        ## Initialize helper values
        prev_idx = -1
        indices_to_exclude = []  # Will be the indices of past dealloc-reallocs

        ## Check each of the executed events in the past
        for event in executed.itertuples():
            ## Get current event
            idx, _, _, curr_type, curr_pttn = event

            ## Skip to next job if previous index is not actually an index,
            ## OR if it is already to be excluded
            if (not prev_idx in executed.index) or (prev_idx in indices_to_exclude):
                prev_idx = idx
                continue

            ## Get event just before the current
            _, _, prev_type, prev_pttn = executed.loc[prev_idx]

            ## If current and previous event operates on the same partition,
            ## AND if they are deallocation and (re)allocation events respectively,
            ## "exclude" them
            if (str(prev_pttn) == str(curr_pttn)) and (
                (prev_type, curr_type) == (EventType.Deallocation, EventType.Allocation)
            ):
                indices_to_exclude.append(prev_idx)
                indices_to_exclude.append(idx)

            ## The "current event" will now be considered as the "previous event"
            prev_idx = idx

        return indices_to_exclude

    def _generate_events(self, jobs, allocs):  # TODO: Refactor?
        ## Create safe copy of PAT
        allocs = allocs.copy()

        ## Initialize event queue; will be a "long" list
        events = pd.DataFrame(columns=self._event_cols)

        ## Initialize helper var for holding latest "execution"
        latest_execution_check = 0

        ## Consider each job, determine their "event" details
        for job in jobs.itertuples():
            ## Unpack job properties
            job_name, job_size, arrival, runtime = job

            ####
            #### "EXECUTION" OF PAST EVENTS
            #### Get up-to-date state of allocations when a job arrives
            ####

            ## Get "executed" events
            executed = events[
                ## Slice from "latest exeuction check" to "current job"
                (latest_execution_check <= events["Time"])
                & (events["Time"] <= arrival)
            ]

            ## Disregard past deallocation-reallocation events
            indices_to_exclude = self._identify_reallocs(executed)
            executed = executed.loc[executed.index.difference(indices_to_exclude)]

            ## Get remaining deallocated partitions
            executed_deallocs = executed[executed["Type"] == EventType.Deallocation]
            pttns_to_dealloc = executed_deallocs["Partition"]

            ## Make these partitions available again
            allocs.loc[pttns_to_dealloc, "Status"] = PartitionStatus.Available

            ## "Move goalpost" lol
            latest_execution_check = arrival

            ####
            #### ARRIVAL
            ####

            ## Add job's ARRIVAL details to "event queue"
            events.loc[len(events)] = (job_name, arrival, EventType.Arrival, pd.NA)

            ####
            #### ALLOCATION
            #### GOAL: Get a [partition name], compute a [starting time]
            ####

            partition = ...
            starting = ...

            ## Get partitions that can hold the job. If none, skip this job.
            can_hold = allocs[allocs["Size"] >= job_size]
            if len(can_hold) == 0:
                ## TODO: Log this in a logfile?
                msg = f"Skipping {job_name} of size {job_size} because it cannot fit in {self.total_allocatable_memory}"
                print(msg)

                continue  # This job will/can not be allocated

            ## Get partitions that can hold the job, and are also available.
            can_hold_and_is_available = can_hold[
                can_hold["Status"] == PartitionStatus.Available
            ]

            ## If may available, kunin yung pinakauna
            if len(can_hold_and_is_available) > 0:

                ## ...kunin yung pinakauna
                candidate_partition = can_hold_and_is_available.iloc[0]

                ## Use that as the [partition name] for this job
                partition = candidate_partition.name

                ## Job will start right when it arrives
                starting = arrival

            ## If walang available, hanapin yung unang matatapos
            else:
                ## Get future events
                future_events = events[arrival < events["Time"]]

                ## Disregard future reallocations
                indices_to_exclude_future = self._identify_reallocs(future_events)
                future_events = future_events.loc[
                    future_events.index.difference(indices_to_exclude_future)
                ]

                ## Filter future event list, kunin lamang ang "deallocations" at "partitions na kayang i-hold ang job"
                pttns_that_can_hold = can_hold.index.to_list()
                candidates = future_events[
                    (future_events["Type"] == EventType.Deallocation)
                    & (future_events["Partition"].isin(pttns_that_can_hold))
                ]

                candidates = candidates.loc[
                    candidates.index.difference(indices_to_exclude)
                ]

                ## Sa pinakaunang event don, kunin yung [partition name] at "deallocation time"
                # candidate_partition_ish = candidates.iloc[0]
                candidate_partition_ish = candidates.iloc[0]
                _, dealloc_time, _, pttn_name = candidate_partition_ish

                ## Use that name as the [partition name] for this job
                partition = pttn_name

                ## Use that "deallocation time" as the [starting time] for this job
                starting = dealloc_time

            ## Add job's ALLOCATION details to "event queue"
            events.loc[len(events)] = (
                job_name,
                starting,
                EventType.Allocation,
                partition,
            )

            ## Change partition status as "allocated"
            ## Should I...?
            allocs.loc[partition, "Status"] = PartitionStatus.Allocated

            ####
            #### DEALLOCATION
            ####

            ## Add job's DEALLOCATION details to "event queue"
            events.loc[len(events)] = (
                job_name,
                starting + runtime,
                EventType.Deallocation,
                partition,
            )

        ## Sort events by time of occurrence, and finally return
        events["_Type_vals"] = events["Type"].apply(lambda row: row.value)
        events = events.sort_values(by=["Time", "_Type_vals"], ignore_index=True)
        events = events.drop(columns="_Type_vals")

        return events


class SummaryTable(NoInstances):
    ## TODO: Remove?
    _summary_table_cols = list(STColumn)

    ## TODO: Compute at instantiation?
    def generate_summary_table(
        self,
        include_arrivals=False,
        normalized=False,
    ):
        # print(self.events)

        ## Reshape event list into summary table
        pivoted = self.events.pivot(
            index="Job No.",
            columns="Type",
            values="Time",
        )

        # print(pivoted)

        ## Drop invalid rows, e.g. for skipped jobs
        pivoted = pivoted.dropna()

        ## Unpack summary table column enumeration
        JobNo, Arrival, Start, Finish, CPUWait, _ = list(STColumn)

        ## Rename index & column headers
        column_renames = {
            pivoted.index.name: JobNo,
            EventType.Arrival: Arrival,
            EventType.Allocation: Start,
            EventType.Deallocation: Finish,
        }
        pivoted = pivoted.reset_index()  # Make job index as column
        pivoted = pivoted.rename(columns=column_renames, errors="raise")
        pivoted.columns.name = None

        ## Add CPU waiting times
        pivoted[CPUWait] = pivoted[Start] - pivoted[Arrival]

        # Revert time normalization, if `normalized = False`
        if not normalized:
            pivoted = self._normalize_time_undo(
                jobs=pivoted,
                columns=[Arrival, Start, Finish],
                base_arrival=self.base_arrival,
            )

        ## Exlude arrival times, if `include_arrivals = False`
        if not include_arrivals:
            pivoted = pivoted.loc[:, ~(pivoted.columns == Arrival)]

        ## Get actual column names, if `normalized = False`
        if not normalized:
            pivoted.columns = [col.value for col in pivoted.columns]

        return pivoted


class TimePoints(NoInstances):

    memory_map_filename_prefix: str
    memory_map_filename_ct: int
    memory_map_filename_dir: str = "memory_map"

    # neutral_color = (
    #     "tab:gray"
    #     if COLOR_DICT == matplotlib.colors.TABLEAU_COLORS
    #     else "gray"  # if COLOR_DICT == matplotlib.colors.CSS4_COLORS
    # )
    neutral_color = "tab:gray"

    def _infer_waiting_jobs(self, event_slice):
        # ## Exclude latest event
        # ## TODO: Exclude only if it is not `EventType.Allocation`
        # event_slice = event_slice.iloc[:-1]

        ## "Summarize" event slice so far
        partial_summary = event_slice.pivot(
            index="Job No.",
            columns="Type",
            values="Time",
        )

        ## Add `EventType.Allocation` column if not present
        Allocation = EventType.Allocation
        if not Allocation in partial_summary.columns:
            partial_summary[Allocation] = pd.NA

        ## Determine job waiting state
        ## Job is currently waiting if it not yet allocated
        ## It can be assumed to have already arrived by this point
        partial_summary[STColumn.IsWaiting.value] = partial_summary[Allocation].isna()

        ## Keep only job name and waiting state
        partial_summary = partial_summary[[STColumn.IsWaiting.value]]

        ## Clean index and column headers
        partial_summary.columns.name = None
        partial_summary = partial_summary.reset_index()

        return partial_summary

    def _update_pat(self, pat, event):
        ## Unpack event props and event column names
        _, job_name, _, event_type, current_partition, _ = event
        Arrival, Allocation, Deallocation = EventType

        ## Short circuit: PAT won't be updated at job arrivals
        if event_type == Arrival:
            return pat

        ## Get partition details
        partition = pat.loc[current_partition, :]
        pttn_name, pttn_size, location, _, _ = partition

        ## Updating behaviors
        if event_type == Allocation:
            ## Change partition status
            pat.at[current_partition, "Status"] = PartitionStatus.Allocated
            pat.at[current_partition, "_job"] = job_name

            ## Get size of event job
            job_size = self.jobs.at[job_name, "Size"]

            ## If job does not fully occupy partition, add "Wasted" partition
            if job_size < pttn_size:
                ## Resize current partition
                pat.at[current_partition, "Size"] = job_size

                ## Determine details of wasted partition
                wasted_size = pttn_size - job_size

                ## Add wasted partition
                new_idx = pat.index.max() + 1
                wasted = PartitionStatus.Wasted
                pat.loc[new_idx] = pttn_name, wasted_size, -1, wasted, job_name

                ## Sort by partition names
                pat = pat.sort_values(by="Partition No.")

        elif event_type == Deallocation:
            ## Change partition status
            pat.at[current_partition, "Status"] = PartitionStatus.Available
            pat.at[current_partition, "_job"] = pd.NA

            ## If partition has "Wasted" part...
            pttn_parts = pat[pat["Partition No."] == pttn_name]
            if (pttn_parts["Status"] == PartitionStatus.Wasted).any():
                ## Wasted partition mask
                wasted_mask = (
                    #
                    (pat["Partition No."] == pttn_name)
                    & (pat["Status"] == PartitionStatus.Wasted)
                    & (pat["_job"] == job_name)
                )

                ## Get wasted size
                wasted_size = pat[wasted_mask]["Size"].item()

                ## Return to actual partition
                pat.at[current_partition, "Size"] += wasted_size

                ## "Drop et jonatan!!!!!"
                wasted_idx = pat.index[wasted_mask].item()
                pat = pat.drop(wasted_idx)

        ## Sort by status precedence
        pat["_Status_vals"] = pat["Status"].apply(lambda row: row.value)
        pat = pat.sort_values(by=["Partition No.", "_Status_vals"])
        pat = pat.drop(columns="_Status_vals")

        ## Update "Location"
        pat["Location"] = pat["Size"].cumsum().shift(periods=1, fill_value=0)

        return pat

    def _clean_pat_proper(self, pat):
        ## Remove "OS partition"
        pat = pat.drop(0)

        ## Distinction for wasted partitions are unnecessary
        pttns = pat["Partition No."].unique()
        for pttn in pttns:
            pat_slc = pat[pat["Partition No."] == pttn]

            if not (len(pat_slc) > 1):
                continue

            ## Unpack indices
            ## Allocated part will be the remaining one
            alloc_idx, *wasted_idx = pat_slc.index

            ## Get total size of partition, store this at Allocated part
            pat.at[alloc_idx, "Size"] = pat_slc["Size"].sum()

            ## Drop other parts
            pat = pat.drop(index=wasted_idx)

        ## Reconstruct "Status" column
        pat["Status"] = [
            f"{status.name} ({job_name})"
            if status == PartitionStatus.Allocated
            else status.name
            for status, job_name in zip(pat["Status"], pat["_job"])
        ]

        ## Remove "_job" column
        column_mask = ~(pat.columns.isin(["_job"]))
        pat = pat[pat.columns[column_mask]]

        return pat

    def _memory_map_label_func(self, row):
        pttn_name, size, _, status, job_name = row

        if pttn_name in ("OS",):  # TODO: Generic name for OS?
            label = pttn_name

        elif status == PartitionStatus.Wasted:
            label = f"{PartitionStatus.Wasted.name}\n({size})"

        elif status == PartitionStatus.Allocated:
            label = f"{job_name} = {size}"

        else:
            label = f"{pttn_name} = {size}"

        return label

    def _generate_random_string(self, length=8):
        return "".join(random.sample(ascii_letters, k=length))

    def _generate_colors(self, pat):
        ## Get list of all available colors
        ## https://matplotlib.org/stable/gallery/color/named_colors.html
        color_names = [name for name in COLOR_DICT.keys() if name != self.neutral_color]

        ## Generate random sequence of colors
        colors = pat[["Partition No."]].copy()
        colors["Color"] = random.sample(color_names, len(colors))

        return colors

    ## TODO: Remove this? Deprecated
    def _save_figure(self, figure, extension="png"):
        ## Create figure directory if it does not exist yet
        if not os.path.isdir(self.memory_map_filename_dir):
            os.mkdir(self.memory_map_filename_dir)

        ## Generate filename
        pref, ct = self.memory_map_filename_prefix, self.memory_map_filename_ct
        filename = f"{pref}_{ct}.{extension}"
        filename = os.path.join(self.memory_map_filename_dir, filename)
        self.memory_map_filename_ct += 1

        ## Save figure
        figure.savefig(filename)

        return filename

    def _save_figure_bytes(self, figure, extension="png"):
        """
        https://stackoverflow.com/questions/63875884/set-a-kivy-image-from-a-bytes-variable
        https://stackoverflow.com/questions/12577110/matplotlib-savefig-to-bytesio-is-slightly-wrong
        https://stackoverflow.com/questions/62513080/unable-to-load-image-from-io-bytesio-in-kivy
        """

        buf = io.BytesIO()

        figure.savefig(buf, format=extension)
        buf.seek(0)

        return (buf, extension)

    def _prepare_memory_map_details(self, pat, base_colors):
        ## Create safe copy
        mmap_sizes = pat.copy()

        ## Create section labels
        mmap_sizes["Label"] = mmap_sizes.apply(self._memory_map_label_func, axis=1)

        ## Assign colors to corresponding partitions
        old_indices = mmap_sizes.index
        mmap_sizes = mmap_sizes.merge(base_colors, on="Partition No.")
        mmap_sizes.index = old_indices

        ## Assign dedicated neutral color to wasted partitions
        mmap_sizes.loc[
            mmap_sizes["Status"] == PartitionStatus.Wasted, "Color"
        ] = self.neutral_color

        ## Keep only relevant columns
        ## TODO: Create enumerations for this?
        mmap_sizes = mmap_sizes[["Size", "Label", "Color"]]

        ## Add "blank" first partition, for proper partition labels
        mmap_sizes = mmap_sizes.reset_index(drop=True)  # Create new order basis
        mmap_sizes.index += 1  # Make room for "blank" partition
        mmap_sizes.loc[0] = 0, "", self.neutral_color  # Add "blank" partition
        mmap_sizes = mmap_sizes.sort_index()  # Move "blank" partition to top

        return mmap_sizes

    def _create_memory_map(self, pat, base_colors, write_file=True):
        ## Add additional details to PAT for memory map construction
        pat = self._prepare_memory_map_details(pat, base_colors)

        ## "Pivot" (transpose) memory map table for stacked plotting
        transposed = pat.T

        ## Create figure
        memory_map = transposed.loc[["Size"]].plot(
            kind="bar",
            stacked=True,
            color=transposed.loc["Color"],
            legend=False,
            figsize=(9, 16),
        )

        ## Reverse y-axis so that plot goes from top to bottom
        memory_map.invert_yaxis()

        ## Hide plot axes and border
        memory_map.axis("off")

        ## Labelling memory sections
        ## https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html
        ## https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
        for idx, container in enumerate(memory_map.containers):
            ## Job names
            memory_map.bar_label(
                container,
                labels=[pat.iloc[idx]["Label"]],
                label_type="center",
                fontweight="bold",
                fontsize="x-large",
            )

            ## Job "sizes" (effective)
            memory_map.bar_label(
                container,
                position=(-144, -5),
            )

        # ## Save memory map figure as image file
        # filename = ""
        # if write_file:
        #     filename = self._save_figure(memory_map.figure)
        # return memory_map.figure, filename

        ## Save memory map as byte file
        bytefile, extension = self._save_figure_bytes(memory_map.figure)
        return memory_map.figure, bytefile, extension

    ## TODO: Rename?
    def generate_timepoint_props(self, write_file=True):
        ## Create safe copies
        events = self.events.copy()
        events["_Type_vals"] = events["Type"].apply(lambda row: row.value)
        events = self._normalize_time_undo(events, ["Time"], self.base_arrival)

        current_pat_complete = self.base_pat_complete.copy()
        current_pat_complete["_job"] = pd.NA

        ## Generate uniform colors
        base_colors = self._generate_colors(current_pat_complete)

        ## Initialize memory map file properties
        self.memory_map_filename_ct = 0
        self.memory_map_filename_prefix = self._generate_random_string()

        ## Timepoint properties: Waiting state, PAT, memory map
        timepoints = []

        ## GUI: Access loading popup reference
        app = MDApp.get_running_app()
        dialog = app.root.current_screen.dialog

        ## Check each individual events
        ## TODO: Iterate through indices only?
        for event in events.itertuples():
            ## Unpack event details
            idx, name, time, event_type, partition, _ = event

            ## Get all events up to current event
            events_slice = events.loc[:idx]

            ####
            #### WAITING STATES
            ####

            waitings = self._infer_waiting_jobs(events_slice)

            ####
            #### PARTITION ALLOCATION TABLE
            ####

            current_pat_complete = self._update_pat(current_pat_complete, event)
            current_pat = self._clean_pat_proper(current_pat_complete)

            ####
            #### MEMORY MAP
            ####
            evtyp = event_type.name.lower()
            prompt_msg = f"Creating memory map for {name} {evtyp}. . ."
            dialog.text = prompt_msg

            # figure, filename = "", ""
            # figure, filename = self._create_memory_map(
            #     current_pat_complete, base_colors, write_file=write_file
            # )

            figure, *bytes_det = self._create_memory_map(
                current_pat_complete, base_colors
            )

            ####
            #### CREATE TIMEPOINT
            ####

            new_timepoint = [
                time,
                name,
                event_type,
                waitings,
                current_pat,
                partition,
                figure,
                bytes_det,
            ]
            timepoints.append(new_timepoint)

            # ####
            # #### ...
            # ####
            # res = [
            #     # event,
            #     # events_slice,
            #     # ####
            #     # self.jobs,
            #     # waitings,
            #     # current_pat_complete,
            #     current_pat,
            #     ####
            # ]
            # print(*res, sep="\n\n", end="\n\n\n")
            # print()

        return (
            timepoints,
            # os.path.join(
            #     self.memory_map_filename_dir,
            #     f"{self.memory_map_filename_prefix}_<Count>.png",
            # ),
            (
                self.memory_map_filename_dir,
                self.memory_map_filename_prefix,
            ),
        )


if __name__ == "__main__":
    for cls_ in (
        NoInstances,
        Preliminaries,
        EventQueue,
        SummaryTable,
        TimePoints,
    ):
        try:
            inst = cls_()
        except NotImplementedError as error:
            print(f"[{cls_}] An error has occurred: {repr(error)}")
