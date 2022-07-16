## TODO: Clean imports
import pandas as pd
from .enumerations import DynamicType, PartitionStatus, EventType, STColumn

import matplotlib
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
    _pttn_tbl_cols = ["Partition No.", "Size", "Location", "Status", "Occupant"]

    def _verify_inputs(self, jobs, total_memory, os_size):
        ## Ensure that all job table columns are as expected
        unexpected = jobs.columns.difference(self._job_table_cols)
        assert (
            not unexpected.any()  # There shouldn't be any unexpected columns
        ), f"Unexpected job table columns: {', '.join(unexpected.to_list())}"

        ## Ensure OS can fit in total memory
        assert (
            total_memory > os_size
        ), f"OS (size {os_size}) cannot fit in memory of size {total_memory}"

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

    def _initialize_partition_table(self, total_memory, os_size):
        ## Initialize general partition table
        gpt = pd.DataFrame(columns=self._pttn_tbl_cols)

        ## Add partition sizes
        allotable = total_memory - os_size
        gpt["Size"] = (os_size, allotable)  # OS is partition 0 (very first)

        ## Infer partition starting locations
        gpt["Location"] = gpt["Size"].cumsum().shift(periods=1, fill_value=0)

        ## Initialize names and statuses
        # gpt.loc[0, ["Partition No.", "Status"]] = ("OS", PartitionStatus.Allocated)
        # gpt.loc[1, ["Partition No.", "Status"]] = ("F1", PartitionStatus.Available)

        gpt["Partition No."] = "OS", ""  # Just derive names of free pttns from status
        gpt["Status"] = PartitionStatus.Allocated, PartitionStatus.Available
        gpt["Occupant"] = "OS", pd.NA

        return gpt

    def _derive_allocation_table(self, status, prefix):
        drv = self.partition_table.loc[1:].copy()  # Exclude OS partition
        drv = drv[drv["Status"] == status]  # Keep only desired statuses
        drv["Partition No."] = prefix + drv.index.astype(str)  # Build partition names

        return drv

    @property
    def pat(self):
        return self._derive_allocation_table(PartitionStatus.Allocated, "P")

    @property
    def fat(self):
        return self._derive_allocation_table(PartitionStatus.Available, "F")


class EventQueue(NoInstances):
    _event_cols = ["Job No.", "Time", "Type"]

    def _combine_continuous_memory(self, memory):
        memory = memory.copy()

        continuous_grp = []

        ## Add dummy end row to ensure loop logic considers all groups
        to_loop = memory.copy()
        to_loop.loc[len(to_loop)] = "", -1, -1, False, pd.NA

        for partition in to_loop.itertuples():
            idx, _, size, loc, availability, occupant = partition

            ## Gather all adjacent partitions in `continuous_grp`
            if availability == PartitionStatus.Available:
                continuous_grp.append(idx)
                continue

            ## As soon as an unavailable partition is encountered,
            ## the previous group of available partitions has ended
            ## Combine them in a single partition

            ## If group is not composed of at least two partitions,
            ## no combination is needed
            if len(continuous_grp) > 1:
                ## Get actual details of partition group
                group = memory.loc[continuous_grp, :]

                ## Get their total size and effective location
                ## Will be located at the position of the first group member
                total = group["Size"].sum()
                base_loc = group["Location"].iloc[0]

                ## Add it as a "new" partition
                memory.loc[len(memory)] = (
                    "",
                    total,
                    base_loc,
                    PartitionStatus.Available,
                    pd.NA,
                )

                ## Drop all partitions in this group
                memory = memory.drop(index=continuous_grp)

                ## Reset properties
                memory = memory.sort_values(by="Location")
                memory = memory.reset_index(drop=True)

            ## Reset group reference
            continuous_grp = []

        return memory

    def _perform_compaction(self, memory, time=None):
        # memory = memory.copy()

        ## Sort first by status (Allocated, Available)
        ## So that all Available partitions will be put to last
        memory["_Status_vals"] = memory["Status"].apply(lambda row: row.value)
        memory = memory.sort_values(by=["_Status_vals", "Location"])
        memory = memory.drop(columns="_Status_vals")

        ## ...recompute locations...
        memory["Location"] = memory["Size"].cumsum().shift(periods=1, fill_value=0)

        ## ...then combine free areas into one
        memory = self._combine_continuous_memory(memory)

        ## Add "Compaction" event to queue
        self.events.loc[len(self.events)] = pd.NA, time, EventType.Compaction

        return memory

    def _execute_deallocations_with_target(self, events, memory, target_size):
        events = events.copy()
        memory = memory.copy()

        ## Consider only Deallocation events
        dealloc_events = events[events["Type"] == EventType.Deallocation]
        dealloc_events = dealloc_events.sort_values(by="Time")

        for event in dealloc_events.itertuples():
            _, job_name, time, event_type = event

            ## Free up memory according with each Deallocation event
            idx_to_free_up = (memory["Occupant"] == job_name).idxmax()
            memory.loc[idx_to_free_up, ["Status", "Occupant"]] = (
                PartitionStatus.Available,
                pd.NA,
            )

            ## Determine if Compaction is possible
            norm_order = memory.reset_index(drop=True).index

            _ = memory.reset_index(drop=True)
            _["_Status_vals"] = _["Status"].apply(lambda row: row.value)
            compaction_order = _.sort_values(by="_Status_vals").index

            ## If free areas are not all adjacent to each other, memory is compactionable
            compactionable = not (compaction_order == norm_order).all()

            if self.with_compaction and compactionable:
                ## If Compaction is enabled and it is possible, perform it
                memory = self._perform_compaction(memory, time=time)

            else:
                ## Else, simply combine possible continuous areas
                memory = self._combine_continuous_memory(memory)
            # print(memory)

            ## Check if enough space has been made
            if (
                memory.loc[memory["Status"] == PartitionStatus.Available, "Size"]
                >= target_size
            ).any():
                return time, memory

        ## If this point is reached, a large enough memory area cannot be determined
        print(memory)
        raise Exception(
            f"Cannot find large enough memory for target {target_size}; max is {memory.loc[1:, 'Size'].max()}"
        )

    def _execute_deallocations(self, events, memory):
        events = events.copy()
        memory = memory.copy()

        ## Consider only Deallocation events
        dealloc_events = events[events["Type"] == EventType.Deallocation]
        dealloc_events = dealloc_events.sort_values(by="Time")

        for event in dealloc_events.itertuples():
            _, job_name, time, event_type = event

            if not (memory["Occupant"] == job_name).any():
                continue

            ## Free up memory according with each Deallocation event
            idx_to_free_up = (memory["Occupant"] == job_name).idxmax()
            memory.loc[idx_to_free_up, ["Status", "Occupant"]] = (
                PartitionStatus.Available,
                pd.NA,
            )

            ## Determine if Compaction is possible
            norm_order = memory.reset_index(drop=True).index

            _ = memory.reset_index(drop=True)
            _["_Status_vals"] = _["Status"].apply(lambda row: row.value)
            compaction_order = _.sort_values(by="_Status_vals").index

            ## If free areas are not all adjacent to each other, memory is compactionable
            compactionable = not (compaction_order == norm_order).all()

            if self.with_compaction and compactionable:
                ## If Compaction is enabled and it is possible, perform it
                memory = self._perform_compaction(memory, time=time)

            else:
                ## Else, simply combine possible continuous areas
                memory = self._combine_continuous_memory(memory)
            # print(memory)

        return memory

    def _get_base_partition(self, candidates):
        ## Select base partition depending on type of dynamic strategy
        if self.dynamic_type == DynamicType.FirstFit:
            ## Use first directly available partition ("first" fit)
            base_pttn = candidates.iloc[0]

        elif self.dynamic_type == DynamicType.BestFit:
            ## Sort candidate partitions by sizes, then use first resulting partition
            ## Will use smallest partition possible ("best" fit)
            candidates = candidates.sort_values(by="Size")
            base_pttn = candidates.iloc[0]

        else:
            ## TODO: Remove this block?
            ## Shouldn't be able to reach this...
            raise Exception

        # ## TODO: Move this line here? Common to each conditionals above
        # base_pttn = candidate_pttns.iloc[0]

        return base_pttn

    def _allocate_job(self, job, base, memory):
        job_name, job_size = job

        idx = base.name
        _, size, loc, *_ = base

        ## Add new partition for the job
        ## Occupy only as much as the job size, insert at base partition's location, set as unavailable
        memory.loc[len(memory)] = (
            "",
            job_size,
            loc,
            PartitionStatus.Allocated,
            job_name,
        )

        ## Reduce base partition by the current job size
        ## Offset base location by current job's size
        memory.at[idx, "Size"] = size - job_size
        memory.at[idx, "Location"] = loc + job_size
        memory = memory.sort_values(by="Location")

        return memory

    def _generate_events(self, jobs, memory):
        ## Create safe copy of partition table
        # memory = memory[1:].copy()
        memory = memory.copy()

        ## Initialize event queue; will be a "long" list
        self.events = pd.DataFrame(columns=self._event_cols)

        ## Initialize reference to latest deallocation/compaction check
        latest_deallocation_check = 0

        ## Consider each job, determine their "event" details
        for job in jobs.itertuples():
            ## Unpack job properties
            job_name, job_size, arrival, runtime = job

            ## Skip jobs that cannot fit in memory
            if job_size > self.total_allocatable_memory:
                ## TODO: Log this in a logfile?
                msg = f"Skipping {job_name} of size {job_size} because it cannot fit in {self.total_allocatable_memory}"
                print(msg)

                continue

            #############
            ## ARRIVAL ##
            #############

            ## Add "Arrival" event
            self.events.loc[len(self.events)] = job_name, arrival, EventType.Arrival

            ################  GOAL: Get a `start` time
            ## ALLOCATION ##
            ################

            start = ...

            #### EXECUTIONS
            #### Get updated state of memory
            #### Produce candidate partitions

            ## Sort events by chronological order
            self.events = self.events.sort_values(by="Time")

            ## Get past events
            events_slice = self.events[
                #
                (latest_deallocation_check <= self.events["Time"])
                & (self.events["Time"] < arrival)
            ]

            ## Execute past events, update memory state
            ## Will also add "Compaction" events accordingly
            memory = self._execute_deallocations(events_slice, memory)

            ## Move "goalpost" lol
            latest_deallocation_check = arrival

            #### CANDIDATE PARTITIONS
            #### Available, and are large enough

            ## Get candidate partitions
            candidate_pttns = memory[
                #
                (memory["Status"] == PartitionStatus.Available)
                & (memory["Size"] >= job_size)
            ]

            ## If there are candidate partitions, use them
            if len(candidate_pttns) != 0:
                ## Job can be started right away
                start = arrival

            ## If there are no candidates, go through future Deallocation events
            else:
                ## Get future events
                events_slice = self.events[
                    #
                    (arrival <= self.events["Time"])
                ]

                ## Determines latest time at which enough memory is freed up for the current job
                ## by performing past Deallocations until enough memory is available
                candidate_start, memory = self._execute_deallocations_with_target(
                    events_slice, memory, target_size=job_size
                )

                ## Move "goalposts"
                latest_deallocation_check = candidate_start

                ## Starting time will either be at job arrival, or time of latest deallocation
                ## whichever is "latest" // last to occur
                start = max(arrival, latest_deallocation_check)

                ## Infer candidate partitions once again
                candidate_pttns = memory[
                    #
                    (memory["Status"] == PartitionStatus.Available)
                    & (memory["Size"] >= job_size)
                ]

            #### BASE PARTITION
            #### Find a base partition, unpack its properties

            base_pttn = self._get_base_partition(candidates=candidate_pttns)

            """
            idx = base_pttn.name
            _, size, loc, *_ = base_pttn
            """

            #### ACTUAL ALLOCATION
            #### Adding to memory, noting it as an event

            """
            ## Add new partition for the job
            ## Occupy only as much as the job size, insert at base partition's location, set as unavailable
            memory.loc[len(memory)] = (
                "",
                job_size,
                loc,
                PartitionStatus.Allocated,
                job_name,
            )

            ## Reduce base partition by the current job size
            ## Offset base location by current job's size
            memory.at[idx, "Size"] = size - job_size
            memory.at[idx, "Location"] = loc + job_size
            memory = memory.sort_values(by="Location")
            """

            memory = self._allocate_job(
                job=(job_name, job_size),
                base=base_pttn,
                memory=memory,
            )

            ## Add "Allocation" event
            self.events.loc[len(self.events)] = job_name, start, EventType.Allocation

            ##################
            ## DEALLOCATION ##
            ##################

            ## Add "Deallocation" event
            self.events.loc[len(self.events)] = (
                job_name,
                start + runtime,
                EventType.Deallocation,
            )

            ##################
            ##              ##
            ##################

            ## Sort events by time of occurrence
            self.events["_Type_vals"] = self.events["Type"].apply(lambda row: row.value)
            self.events = self.events.sort_values(
                by=["Time", "_Type_vals"], ignore_index=True
            )
            self.events = self.events.drop(columns="_Type_vals")

            # print(arrival, latest_deallocation_check)
            # print(memory)
            # print(self.events)
            # print("-" * 32)

        ## Check for possible remaining Compaction events
        ## Inspect events from last check, up to very last event
        events_slice = self.events[
            #
            (latest_deallocation_check <= self.events["Time"])
        ]

        ## Perform remaining deallocations
        ## Will add Compaction events accordingly
        memory = self._execute_deallocations(events_slice, memory)

        ## Sort events by time of occurrence
        self.events["_Type_vals"] = self.events["Type"].apply(lambda row: row.value)
        self.events = self.events.sort_values(
            by=["Time", "_Type_vals"], ignore_index=True
        )
        self.events = self.events.drop(columns="_Type_vals")

        # print(memory)
        # print(self.events)

        return self.events


class SummaryTable(NoInstances):
    ## TODO: Remove?
    _summary_table_cols = list(STColumn)

    ## TODO: Compute at instantiation?
    def generate_summary_table(
        self,
        include_arrivals=False,
        normalized=False,
    ):
        ## Create safe local copy
        events = self.events.copy()

        ## Exclude Compaction events from consideration
        events = events[~(events["Type"] == EventType.Compaction)]

        ## Reshape event list into summary table
        pivoted = events.pivot(
            index="Job No.",
            columns="Type",
            values="Time",
        )

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
    memory_map_filename_ct: int
    memory_map_filename_prefix: str
    memory_map_filename_dir: str = "memory_map"

    neutral_color = "tab:gray"

    @property
    def color_selection(self):
        return [name for name in COLOR_DICT.keys() if name != self.neutral_color]

    def _generate_random_string(self, length=8):
        return "".join(random.sample(ascii_letters, k=length))

    def _infer_waiting_jobs(self, event_slice):
        # ## Exclude latest event
        # ## TODO: Exclude only if it is not `EventType.Allocation`
        # event_slice = event_slice.iloc[:-1]

        ## Exclude Compaction events from consideration
        event_slice = event_slice[~(event_slice["Type"] == EventType.Compaction)]

        ## "Summarize" event slice so far
        ptl_summ = event_slice.pivot(
            index="Job No.",
            columns="Type",
            values="Time",
        )

        ## Add `EventType.Allocation` column if not present
        Allocation = EventType.Allocation
        if not Allocation in ptl_summ.columns:
            ptl_summ[Allocation] = pd.NA

        ## Determine job waiting state
        ## Job is currently waiting if it not yet allocated
        ## It can be assumed to have already arrived by this point
        ptl_summ[STColumn.IsWaiting.value] = ptl_summ[Allocation].isna()

        ## Keep only job name and waiting state
        ptl_summ = ptl_summ[[STColumn.IsWaiting.value]]

        ## Clean index and column headers
        ptl_summ.columns.name = None
        ptl_summ = ptl_summ.reset_index()

        ## Transpose to wide format
        ptl_summ = ptl_summ.T
        ptl_summ = ptl_summ.rename(columns=ptl_summ.loc["Job No.", :])
        ptl_summ = ptl_summ.drop(index="Job No.")

        return ptl_summ

    def _update_alloc_table(self, memory, event):
        ## Unpack event properties and possible types
        (
            _,
            job_name,
            _,
            event_type,
        ) = event
        Deallocation, Compaction, Arrival, Allocation = EventType
        Allocated, Available = PartitionStatus

        ## SHORT CIRCUIT: Table does not need to be updated at job arrivals
        if event_type == Arrival:
            return memory

        ## Save area colors
        _colors = memory[["Occupant", "Color"]].copy()
        _colors = _colors.drop_duplicates(
            subset="Occupant",
            ignore_index=True,
        )
        memory = memory.drop(columns="Color")

        ## Behaviors for updating
        if event_type == Allocation:
            ## Retrieve job size
            job_size = self.jobs.at[job_name, "Size"]

            ## Get a base partition
            avl_pttns = memory[
                #
                (memory["Status"] == Available)
                & (memory["Size"] >= job_size)
            ]
            base_pttn = self._get_base_partition(candidates=avl_pttns)

            ## Perform allocation
            memory = self._allocate_job(
                job=(job_name, job_size),
                base=base_pttn,
                memory=memory,
            )

            ## Add area colors back
            # memory["Color"] = _colors
            memory = memory.merge(_colors, on="Occupant", how="left")

            ## Add new color for newly allocated area
            _colors = _colors["Color"].to_list()
            new_sel = [c for c in self.color_selection if c not in _colors]
            new_areas = memory[memory["Color"].isna()]
            # print(new_sel)
            # print(new_areas)
            memory.loc[
                memory["Color"].isna(),
                "Color",
            ] = random.sample(new_sel, k=len(new_areas))

        elif event_type == Deallocation:
            idx_to_free_up = (memory["Occupant"] == job_name).idxmax()
            memory.loc[idx_to_free_up, ["Status", "Occupant"]] = (
                PartitionStatus.Available,
                pd.NA,
            )

            memory = self._combine_continuous_memory(memory)

            ## Add area colors back
            # memory["Color"] = _colors
            memory = memory.merge(_colors, on="Occupant", how="left")

            ## Set available areas to neutral color
            memory.loc[
                memory["Status"] == PartitionStatus.Available, "Color"
            ] = self.neutral_color

        elif event_type == Compaction:
            memory = self._perform_compaction(memory)

            ## Add area colors back
            # memory["Color"] = _colors
            memory = memory.merge(_colors, on="Occupant", how="left")

        return memory

    def _pat_status_func(self, row):
        if row.empty:
            return ""

        pttn_name, _, _, status, occupant, _ = row
        Allocated, Available = PartitionStatus

        if status == Allocated:
            status = f"Alloc. ({occupant})"

        # elif status == Available:
        #     ...

        return status

    def _infer_pat_fat(self, memory):
        ## Exclude OS partition
        slc = memory.iloc[1:]

        ## Unpack statuses
        Allocated, Available = PartitionStatus

        memory_slices = []
        for status, prefix, col in (
            (Allocated, "P", "Partition #"),  # PAT
            (Available, "F", "FA #"),  # FAT
        ):
            ## Get indices
            idxs = slc[slc["Status"] == status].index

            ## Add Partition No. // Names
            idxs_reset = pd.RangeIndex(stop=len(idxs))
            idxs_reset += 1
            memory.loc[idxs, "Partition No."] = prefix + idxs_reset.astype(str)

            mem_slice = memory.copy()

            ## Rename column, statuses
            mem_slice = mem_slice.rename(columns={"Partition No.": col})
            mem_slice["Status"] = mem_slice.apply(self._pat_status_func, axis=1)

            ## Exclude irrelevant columns
            col_mask = ~mem_slice.columns.isin(["Color", "Occupant"])
            mem_slice = mem_slice.loc[idxs, col_mask]

            ## Store slice
            memory_slices.append(mem_slice)

        return memory_slices

    def _memory_map_label_func(self, row):
        pttn_name, size, *_ = row

        if pttn_name in ("OS",):  # TODO: Generic name for OS?
            label = pttn_name

        else:
            label = f"{pttn_name} = {size}"

        return label

    def _prepare_memory_map_details(self, memory):
        ## Create section labels
        memory["Label"] = memory.apply(self._memory_map_label_func, axis=1)

        ## Keep only relevant columns
        ## TODO: Create enumerations for this?
        memory = memory[["Label", "Size", "Color"]]

        ## Add "blank" first partition, for proper partition labels
        memory = memory.reset_index(drop=True)  # Create new order basis
        memory.index += 1  # Make room for "blank" partition
        memory.loc[0] = "", 0, self.neutral_color  # Add "blank" partition
        memory = memory.sort_index()  # Move "blank" partition to top

        return memory

    def _save_figure(self, figure):
        ## Create figure directory if it does not exist yet
        if not os.path.isdir(self.memory_map_filename_dir):
            os.mkdir(self.memory_map_filename_dir)

        ## Generate filename
        pref, ct = self.memory_map_filename_prefix, self.memory_map_filename_ct
        filename = f"{pref}_{ct}.png"
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

    def _create_memory_map(self, memory):
        memory = memory.copy()

        ## Add additional details to PAT for memory map construction
        memory = self._prepare_memory_map_details(memory)

        ## "Pivot" (transpose) memory map table for stacked plotting
        transposed = memory.T

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
                labels=[memory.iloc[idx]["Label"]],
                label_type="center",
                fontweight="bold",
                fontsize="x-large",
            )

            ## Job "sizes" (effective)
            memory_map.bar_label(
                container,
                position=(-144, -5),
            )

        """
        ## Save memory map figure as image file
        filename = self._save_figure(memory_map.figure)
        return memory_map.figure, filename
        """

        ## Save memory map as byte file
        bytefile, extension = self._save_figure_bytes(memory_map.figure)
        return memory_map.figure, bytefile, extension

    def generate_timepoint_props(self):
        ## Create safe copies
        events = self.events.copy()
        # events["_Type_vals"] = events["Type"].apply(lambda row: row.value)
        events = self._normalize_time_undo(events, ["Time"], self.base_arrival)

        gpt = self.partition_table.copy()

        ## Initialize colors
        ## TODO: Cannot be known beforehand? Generate color per new partition created
        gpt["Color"] = random.sample(self.color_selection, k=len(gpt))
        gpt.loc[
            gpt["Status"] == PartitionStatus.Available, "Color"
        ] = self.neutral_color

        ## Initialize memory map file properties
        self.memory_map_filename_ct = 0
        self.memory_map_filename_prefix = self._generate_random_string()

        ## GUI: Access loading popup reference
        app = MDApp.get_running_app()
        dialog = app.root.current_screen.dialog

        ## Timepoint properties: Waiting state, PAT, FAT, memory map
        timepoints = []

        ## Check each individual events
        for event in events.itertuples():
            ## Unpack event details
            (idx, job_name, time, event_type) = event
            # print(time, job_name, event_type, sep=" | ")

            ## Get all events up to current event
            events_slice = events.loc[:idx]
            # print(events_slice)

            ###################
            ## WATING STATES ##
            ###################
            waitings = self._infer_waiting_jobs(events_slice)
            # print(waitings)

            ###############
            ## PAT | FAT ##
            ###############
            gpt = self._update_alloc_table(gpt, event)
            pat, fat = self._infer_pat_fat(gpt)
            # print(gpt)
            # print(pat)
            # print(fat)

            ################
            ## MEMORY MAP ##
            ################
            pholder = "" if pd.isna(job_name) else f"{job_name} "
            msg = f"Creating memory map for {pholder}{event_type.name.lower()}. . ."
            dialog.text = msg

            # figure, filename = self._create_memory_map(memory=gpt)
            figure, *bytes_det = self._create_memory_map(memory=gpt)

            ###############
            ## TIMEPOINT ##
            ###############
            new_tpt = [
                time,
                job_name,
                event_type,
                waitings,
                pat,
                fat,
                figure,
                bytes_det,
            ]
            timepoints.append(new_tpt)

        """
        filename_struct = os.path.join(
            self.memory_map_filename_dir,
            f"{self.memory_map_filename_prefix}_<Count>.png",
        )
        return (
            timepoints,  # Properties per each timepoint
            filename_struct,  # Base filename of memory map files
        )
        """

        base_struct = (
            self.memory_map_filename_dir,
            self.memory_map_filename_prefix,
        )
        return (
            timepoints,
            base_struct,
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
