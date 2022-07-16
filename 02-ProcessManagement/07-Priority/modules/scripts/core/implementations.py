from .enumerations import SchedulingType, EventType, ProcessState, ProcessColumns
import pandas as pd

import os
import io
import random
import string

## For logging purposes
# import logging
from kivy.logger import Logger

## These are submodules of `matplotlib` and are thus must be imported separately
import matplotlib.axes
import matplotlib.figure
import matplotlib.colors
import matplotlib.text


class Preliminaries:
    validations = {
        # "processes": {"MAX_COUNT": 5},
        "processes": {"MAX_COUNT": 10},
        "scheduling": list(SchedulingType),
        "columns": list(ProcessColumns),
    }

    def __init__(self, processes, scheduling_type):
        ## Check processes
        self.validate_processes(processes)
        processes = self.prepare_processes(processes)

        ## Check scheduling type
        self.validate_scheduling(scheduling_type)

        ## Assign inputs as attributes
        self.processes = processes
        self.scheduling_type = scheduling_type

    def validate_processes(self, processes):
        ## Condition: Maximum of five (5) processes only
        threshold = self.validations["processes"]["MAX_COUNT"]
        msg = f"Maximum process count set to {threshold}; got {len(processes)}."
        assert len(processes) <= threshold, msg

        ## Condition: All columns are supported
        columns = " | ".join([str(col) for col in ProcessColumns][:3])
        msg = f"Must be one of: {columns}"
        condition = processes.columns.isin(self.validations["columns"]).all()
        assert condition, msg

        ## ... (Other validation tests)

    def prepare_processes(self, processes):
        ## Sort by chronological order for proper simulation
        processes = processes.sort_values(by=ProcessColumns.Arrival)

        ## Set "Process" column as index
        processes = processes.set_index(ProcessColumns.Process)

        ## Add new columns
        # Start, Finish = ProcessColumns.Start, ProcessColumns.Finish
        _, _, _, Priority, Start, Finish, Turnaround, Waiting, _ = ProcessColumns
        processes[[Start, Finish, Turnaround, Waiting]] = -1

        """
        ## Drop Priority column
        if Priority in processes.columns:
            processes = processes.drop(columns=Priority)
        """

        return processes

    def validate_scheduling(self, input_type):
        ## Condition: Scheduling type must be supported
        supported = self.validations["scheduling"]
        msg = f"Scheduling type must be one of {supported}; got {input_type}."
        assert input_type in supported, msg

        ## ... (Other validation tests)


"""
class RoundRobin:  # Mixin for clarity
    rr_quantum: int
    queue: list[tuple]  # (name: str, remaining_time: int, previously_ran: bool)

    def __init__(self, *args, **kwargs):
        msg = f"`{self.__class__.__name__}` is intended only as a mixin!"
        raise NotImplementedError(msg)

    def rr_enqueue(self, proc_name, burst, is_suspended=False):
        self.queue.append((proc_name, burst, is_suspended))

    def rr_dequeue(self):
        return self.queue.pop(0)
"""


class Processor:
    process = {  ## TODO: Better way?
        "Name": None,
        "Runtime": None,
        "Elapsed": None,  # TODO: Remove?
        "Priority": None,
    }

    finished: int = 0  # Count of processes finished
    queue: pd.DataFrame
    queue_cols: list[ProcessColumns] = list(ProcessColumns)[:4] + ["Suspended"]

    ## TODO: Remove this?
    @property
    def is_idle(self) -> bool:
        ## Processor is idling if there is no process name
        return self.process["Name"] is None

    def __init__(self):
        self.queue = pd.DataFrame(columns=self.queue_cols)

    def dispatch(self, proc_name, runtime, priority):
        msg = "Processor must be idle by this point; something has gone wrong"
        assert self.is_idle, msg

        self.process["Name"] = proc_name
        self.process["Runtime"] = runtime
        # self.process["Elapsed"] = 1
        self.process["Priority"] = priority

    def release(self, from_preemption=False):
        ## Reset process props
        # self.process["Name"] = None
        # self.process["Runtime"] = 0
        # self.process["Elapsed"] = 1

        for key in self.process:
            self.process[key] = None

        ## Increment "finished processes" counter
        if not from_preemption:
            self.finished += 1

    """ Moved to simulation, for event tracking
    def preempt(self):
        ## Enqueue current process again
        proc_name, runtime, *_ = self.process.values()
        self.rr_enqueue(proc_name, runtime)

        ## Release process from processor
        self.release(from_preemption=True)

        ## Get first process from queue
        proc_name, runtime = self.rr_dequeue()

        ## Assign dequeued process to processor
        self.dispatch(proc_name, runtime)
    """

    def processing(self):
        ## SHORT: While time quantum or process has not elapsed...
        if not self.process["Elapsed"] in (self.rr_quantum, self.process["Runtime"]):
            self.process["Elapsed"] += 1
            return

        ## Reduce burst value
        self.process["Runtime"] -= self.process["Elapsed"]
        assert self.process["Runtime"] >= 0, "Negative runtime; something went wrong"

        event = None

        if self.process["Runtime"] == 0:  # For releasing
            # self.release()
            event = EventType.Release
        else:  # For preemption (Process not yet finished)
            # self.preempt()
            event = EventType.Suspend

        return event

    def enqueue(self, proc, rem_time, queue_time, prio_level, from_suspension):
        q = self.queue
        q.loc[len(q)] = (proc, rem_time, queue_time, prio_level, from_suspension)

        # ## DEBUG: View queue with every enqueue
        # print(f"\n{q}\n")

    def dequeue(self):
        row0 = self.queue.iloc[0]
        self.queue = self.queue.iloc[1:].reset_index(drop=True)
        return row0


class Simulation:

    ## Simulation time properties initialized in "actual" simulation
    clock: int
    end: int  # Running value; changed with each process dispatch

    ## Data
    processor: Processor
    processes: pd.DataFrame
    processes_full: pd.DataFrame  # Will have idles first
    idle_label = "<idle>0"

    ## For creating an [event | timepoint] list
    events: pd.DataFrame
    events_cols = ["Process", "Time", "Type"]
    init_label = "<init>"
    event_state_map = {
        EventType.Release: ProcessState.Terminated,
        EventType.Arrival: ProcessState.Ready,
        EventType.Dispatch: ProcessState.Running,
        EventType.Idling: ProcessState.Idle,
        EventType.Suspend: ProcessState.Blocked,
        EventType.Resume: ProcessState.Running,
    }

    timepoints: list[list]
    states: pd.DataFrame
    states_idxs = ["State"]

    ## Derived values
    CPU_util: float
    ATA: float
    AWT: float

    ## DEBUG: Debugging props
    DEBUG = False
    DEBUG_HARD_END = 100

    def __init__(self, processes: pd.DataFrame, scheduling_type: SchedulingType):
        ## Assign instance attributes
        self.processor = Processor()
        self.processes = processes
        self.scheduling = scheduling_type

        ## Start simulation
        ## TODO: Remove this? Start explicitly?
        self.simulate()

        ## Add post-mortem values
        self.finalize()

    def __event_add(self, process, time, event_type):
        self.events.loc[len(self.events)] = process, time, event_type
        time = self.clock
        Logger.info(f"Event - {time=} {process=} {str(event_type)=}")

        ## Blank out idle period fot states and timepoints
        if event_type == EventType.Idling:
            process = ""

        ## Update process states
        if event_type in self.event_state_map:
            state = self.event_state_map[event_type]
            self.states.loc["State", process] = state
            # self.states.loc["Burst", process] = burst

        ## Set blocked processes
        ready_mask = self.states.iloc[0] == ProcessState.Ready
        ready_procs = self.states.columns[ready_mask]

        Arrival = ProcessColumns.Arrival
        blocked_mask = self.processes.loc[ready_procs, Arrival] < self.clock
        blocked_procs = ready_procs[blocked_mask]

        self.states.loc["State", blocked_procs] = ProcessState.Blocked
        # self.states.loc["Burst", blocked_procs] = burst

        ## Add timepoint
        """ Time | Process name | Event type | Process states | Figure (Constructed separately!) """
        # timepoint = [time, process, event_type, self.states.copy(), ...]
        timepoint = [time, process, event_type, self.states.copy()]
        self.timepoints.append(timepoint)

    def __apply_scheduling(self, processes):
        """
        Apply scheduling logic
            FCFS - No additional side effect
            SJF  - Sort by sizes
            Prio - ...
            ...
        """

        FCFS, SJF, PrioritySch, SRTF, RR = SchedulingType
        Process, Burst, Arrival, PriorityCol, *_ = ProcessColumns

        ## Reset index as column
        index_name = processes.index.name or "index"
        processes = processes.reset_index()

        if self.scheduling == FCFS:
            ## First come (First to arrive)
            processes = processes.sort_values(by=[Arrival, index_name])

        elif self.scheduling == SJF:
            processes = processes.sort_values(by=[Burst, index_name])

        elif self.scheduling == PrioritySch:
            processes = processes.sort_values(by=[PriorityCol, index_name])

        # elif self.scheduling == SRTF:
        #     ...

        # elif self.scheduling == RR:
        #     ...

        else:
            raise NotImplementedError(
                f"Type '{self.scheduling.value}'' not yet implemented."
            )

        ## Set index again
        processes = processes.set_index(index_name)

        return processes

    def __sim_arrivals(self):
        procs = self.processes
        cpu = self.processor

        """
        ## Sort for idle periods
        label = procs.iloc[-1].name
        start = procs.at[label, ProcessColumns.Start]
        procs.loc[label, ProcessColumns.Finish] = self.clock
        procs.loc[label, ProcessColumns.Burst] = self.clock - start
        self.processes = self.processes.sort_values(by=ProcessColumns.Arrival)
        """

        ## Get arriving processes
        Arrival = ProcessColumns.Arrival
        arrivals = procs.loc[procs[Arrival] == self.clock].copy()
        arrivals = self.__apply_scheduling(arrivals)

        ## SHORT: Do nothing if no jobs arrived
        if len(arrivals) == 0:
            return

        ## Create running iterable
        iterable = arrivals.itertuples()

        ## Get "first" process
        ## Also removes it from running iterable
        candidate = next(iterable)
        name_idx, burst, arrival, priority_level, *_ = candidate
        self.__event_add(name_idx, self.clock, EventType.Arrival)

        if cpu.is_idle:
            ## If nothing is currently being processed, dispatch candidate process to CPU
            """
            self.__cpu_proc_dispatch(
                proc_name=name_idx,
                start=self.clock,
                runtime=burst,
            )
            """
            cpu.dispatch(name_idx, burst, priority_level)
            self.__event_add(name_idx, self.clock, EventType.Dispatch)
            procs.at[name_idx, ProcessColumns.Start] = self.clock

        elif priority_level < cpu.process["Priority"]:
            """
            If candidate process is of higher priority,
            (Level 1 is higher than 2, etc., so logically "less")
            then preempt current process
            """

            ## Enqueue current process again
            curr_proc, curr_burst, _, curr_prio = cpu.process.values()
            # cpu.rr_enqueue(proc_name, runtime, is_suspended=True)
            cpu.enqueue(curr_proc, curr_burst, self.clock, curr_prio, True)

            ## Release process from processor
            cpu.release(from_preemption=True)
            self.__event_add(curr_proc, self.clock, EventType.Suspend)

            ## Dispatch candidate process
            cpu.dispatch(name_idx, burst, priority_level)
            self.__event_add(name_idx, self.clock, EventType.Dispatch)
            procs.at[name_idx, ProcessColumns.Start] = self.clock

        else:
            ## Candidate process cannot be accommodated, and must be enqueue for later
            cpu.enqueue(name_idx, burst, self.clock, priority_level, False)

        ## Other processes that have arrived will be enqueued for later consideration
        for proc in iterable:
            name_idx, burst, arrival, priority_level, *_ = candidate
            self.__event_add(name_idx, arrival, EventType.Arrival)
            cpu.enqueue(name_idx, burst, self.clock, priority_level, False)

    def __sim_idle(self):
        procs = self.processes_full
        cpu = self.processor

        if not cpu.is_idle:
            if self.idle_label in procs.index:
                ## Compute idle periods
                Start, Finish = [ProcessColumns.Start, ProcessColumns.Finish]
                procs.at[self.idle_label, Finish] = self.clock - 1  # Off-by-one
                start, finish = procs.loc[self.idle_label, [Start, Finish]]
                procs.at[self.idle_label, ProcessColumns.Burst] = finish - start

                ## Move forward idle label
                idx = procs.index.str.contains(self.idle_label).sum()
                self.idle_label = self.idle_label[:-1] + str(idx)
            return

        if self.idle_label not in procs.index:
            procs.loc[self.idle_label] = -1, self.clock, 0, self.clock, -1, -1, -1

        if not self.events["Process"].str.contains(self.idle_label).any():
            self.__event_add(self.idle_label, self.clock, EventType.Idling)

    def __sim_proc_execute(self):
        cpu = self.processor
        procs = self.processes

        ## SHORT: Nothing to execute if processor is idle
        if cpu.is_idle:
            return

        cpu.process["Runtime"] -= 1

        ## SHORT: If process still is not finished, continue simulation
        if cpu.process["Runtime"] > 0:
            return

        ## Release process
        name_idx = cpu.process["Name"]
        self.__event_add(name_idx, self.clock, EventType.Release)
        cpu.release()
        procs.at[name_idx, ProcessColumns.Finish] = self.clock

        ## If queue is not empty...
        if len(cpu.queue) > 0:
            ## Get first process from queue
            cpu.queue = self.__apply_scheduling(cpu.queue)
            proc_name, runtime, _, priority_level, is_suspended = cpu.dequeue()

            ## Assign dequeued process to processor
            cpu.dispatch(proc_name, runtime, priority_level)

            if is_suspended:
                self.__event_add(proc_name, self.clock, EventType.Resume)
            else:
                self.__event_add(proc_name, self.clock, EventType.Dispatch)
                procs.at[proc_name, ProcessColumns.Start] = self.clock

        """
        event = cpu.processing()

        ## SHORT: Run the rest of the method only for these events
        if event not in (EventType.Release, EventType.Suspend):
            return

        if event == EventType.Release:
            name_idx = cpu.process["Name"]
            self.__event_add(name_idx, self.clock, EventType.Release)
            cpu.release()
            procs.at[name_idx, ProcessColumns.Finish] = self.clock

        elif event == EventType.Suspend:  # Preemption logic
            ## Enqueue current process again
            proc_name, runtime, *_ = cpu.process.values()
            cpu.rr_enqueue(proc_name, runtime, is_suspended=True)

            ## Release process from processor
            cpu.release(from_preemption=True)
            self.__event_add(proc_name, self.clock, EventType.Suspend)

        ## If queue is not empty...
        if len(cpu.queue) > 0:
            ## Get first process from queue
            proc_name, runtime, is_suspended = cpu.rr_dequeue()

            ## Assign dequeued process to processor
            cpu.dispatch(proc_name, runtime)

            if is_suspended:
                self.__event_add(proc_name, self.clock, EventType.Resume)
            else:
                self.__event_add(proc_name, self.clock, EventType.Dispatch)
                procs.at[proc_name, ProcessColumns.Start] = self.clock
        """

    def simulate(self):
        ## Initialize simulation properties
        self.clock = 0
        self.end = -1

        self.events = pd.DataFrame(columns=self.events_cols)
        self.timepoints = list()
        self.states = pd.DataFrame(index=self.states_idxs)

        self.processes_full = pd.DataFrame(columns=self.processes.columns)
        self.processes_full.index.name = self.processes.index.name

        ## "Before time 0..." | Initial state
        self.__event_add(self.init_label, -1, EventType.Init)

        ## "Simulation" proper
        while self.processor.finished < len(self.processes):
            Logger.info(f"{self.clock=}")

            ## Consider idle times
            self.__sim_idle()

            ## "Execute" process
            self.__sim_proc_execute()

            ## Consider arriving processes
            self.__sim_arrivals()

            ## DEBUG: Hard break to prevent infinite loop
            if self.DEBUG and self.clock == self.DEBUG_HARD_END:
                Logger.warning("Debug hard break")
                break

            # print(f"{self.clock:>3} {self.processor.queue}")

            ## Move forward in time
            self.clock += 1

        # breakpoint()

    def finalize(self):
        procs = self.processes
        procs_full = self.processes_full

        ## Compute Turnaround and Waiting times
        _, Burst, Arrival, _, Start, Finish, Turnaround, Waiting, _ = ProcessColumns
        procs[Turnaround] = procs[Finish] - procs[Arrival]

        # procs[Waiting] = procs[Start] - procs[Arrival]
        _, _, _, Arrival, Dispatch, Suspend, Resume = EventType
        for proc in procs.index:  # AWT calc. didferent due to preemption
            proc_evs = self.events[self.events["Process"] == proc]
            starts = proc_evs.loc[proc_evs["Type"].isin([Dispatch, Resume]), "Time"]
            stops = proc_evs.loc[proc_evs["Type"].isin([Arrival, Suspend]), "Time"]

            procs.at[proc, Waiting] = starts.sum() - stops.sum()

        ## Compute derived values
        idle_time = procs_full[Burst].sum()  # "Full" only has idle periods
        duration = procs[Finish].max()
        self.CPU_util = (1 - (idle_time / duration)) * 100

        self.ATA = procs[Turnaround].mean()
        self.AWT = procs[Waiting].mean()
        # breakpoint()

        ## DEBUG: Delete these! For use in `breakpoint()`
        [self.CPU_util, self.ATA, self.AWT]
        self.events[self.events["Type"].isin([EventType.Dispatch, EventType.Resume])]
        # breakpoint()

        ## Relabel idle periods
        idle_mask = self.events["Type"] == EventType.Idling
        self.events.at[idle_mask, "Process"] = self.idle_label[:-1]

        ## Add idle periods to processes
        self.processes_full = pd.concat([self.processes, self.processes_full])
        self.processes_full = self.processes_full.sort_values(by=ProcessColumns.Start)


class Plotting:
    neutral_color = "tab:gray"  # From `matplotlib`'s Tableau colors
    idle_label = "<idle>"  # `Simulation` has similar attrib.; does not have to be same
    trunc_thresh = 1 / 20  # Minimum proportion to truncate bar labels
    figure_size = (21, 3)
    directory: str = "gantt"

    struct: pd.DataFrame
    gantt: matplotlib.axes._subplots.Subplot
    figure: matplotlib.figure.Figure
    annotations: list[matplotlib.text.Annotation]

    visible_bars: int = 0  # Number of currently visible bars in the plot

    @property  # Cannot define this as a class attribute...
    def color_selection(self):
        colors = matplotlib.colors.TABLEAU_COLORS.keys()
        return [name for name in colors if name != self.neutral_color]

    @property  # Shorthand
    def figure(self):
        return self.gantt.figure

    def __init__(self, events):
        ## Prepare representative structure
        self.struct = self.__prepare_repr_struct(events)

        ## Draw bar plot
        self.__draw_plot()

    def __prepare_repr_struct(self, events):
        events = events.copy()

        ## Get last event time
        last_time = events["Time"].max()

        ## Generate effective "process list"
        ## Because of preemption, simple process list cannot be used
        event_types = (EventType.Dispatch, EventType.Resume, EventType.Idling)
        events = events[events["Type"].isin(event_types)].copy()

        events["Next"] = events["Time"].shift(periods=-1, fill_value=last_time)
        events["Burst"] = events["Next"] - events["Time"]

        ## Initialize colors column
        colors = {"Process": events["Process"].unique()}
        colors["Color"] = random.sample(self.color_selection, k=len(colors["Process"]))
        colors = pd.DataFrame(colors)
        events = events.merge(colors, on="Process")

        ## Set idle colors as neutral
        idle_mask = events["Process"].str.contains(self.idle_label)
        events.at[idle_mask, "Color"] = self.neutral_color

        ## Sort events by chronological order
        events = events.sort_values(by="Time", ignore_index=True)

        ## Perform label truncation accordingly
        rel_pcts = events["Burst"] / events["Burst"].sum()

        events["pcts"] = rel_pcts
        candidate_mask = events["pcts"] < self.trunc_thresh
        events.loc[candidate_mask, "Process"] = ""  # "Remove" by setting as blank
        events = events.drop(columns="pcts")

        ## Reset index counts
        events = events.reset_index(drop=True)

        ## Keep only relevant columns
        events = events[["Process", "Burst", "Color"]]

        ## Add first blank bar, for helping bar labels
        events.index += 1
        events.loc[0] = "", 0, self.neutral_color
        events = events.sort_index()

        ## "Pivot", for stacked plotting
        events = events.T

        return events

    def __draw_plot(self):
        ## Plot Gantt chart
        gantt = self.struct.loc[["Burst"]].plot(
            kind="barh",
            stacked=True,
            color=self.struct.loc["Color"],
            legend=False,
            figsize=self.figure_size,
        )

        ## Reduce white spaces
        gantt.axis("tight")

        ## Hide plot axes and border
        gantt.axis("off")

        ## Labelling memory sections
        ## https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html
        ## https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
        for idx, container in enumerate(gantt.containers):
            ## Process names
            gantt.bar_label(
                container,
                labels=[self.struct.at["Process", idx]],
                label_type="center",
                fontweight="bold",
                fontsize="x-large",
                # rotation="vertical",
            )

            ## Process endtimes
            gantt.bar_label(
                container,
                position=(-5, -85),
            )

        ## Assign as attribute
        self.gantt = gantt
        self.annotations = [
            child
            for child in self.gantt.get_children()
            if isinstance(child, matplotlib.text.Annotation)
        ]

    def __generate_random_string(self, length=8):
        return "".join(random.sample(string.ascii_letters, k=length))

    def toggle_visibility(self):
        self.visible_bars = 0

        for idx, container in enumerate(self.gantt.containers):
            ## Hide bars
            container.patches[0].set_alpha(0)

            ## Hide text labels
            self.annotations[(idx * 2) + 0].set_alpha(0)
            self.annotations[(idx * 2) + 1].set_alpha(0)

    def increment_visibility(self):
        if self.visible_bars >= len(self.gantt.containers):
            return

        idx = self.visible_bars
        container = self.gantt.containers[idx]

        container.patches[0].set_alpha(1)
        self.annotations[idx * 2].set_alpha(1)
        self.annotations[(idx * 2) + 1].set_alpha(1)

        self.visible_bars += 1

    def save(self, filename, extension="png"):
        # self.figure.savefig(filename, format=extension)
        self.figure.savefig(filename)
        return f"{filename}.{extension}"

    def save_all(self):
        ## Create figure directory if it does not exist yet
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        ## Hide all sections initially
        self.toggle_visibility()

        ## Save figures with incremental visibility
        basename = self.__generate_random_string()

        for idx, container in enumerate(self.gantt.containers):
            self.increment_visibility()

            filename = f"{basename}_{idx}"
            filename = os.path.join(self.directory, filename)
            self.save(filename)

    def save_bytes(self, extension="png"):
        buf = io.BytesIO()
        self.gantt.figure.savefig(buf, format=extension)
        buf.seek(0)
        return (buf, extension)

    def save_bytes_all(self):
        self.toggle_visibility()

        byte_details = []
        for idx, container in enumerate(self.gantt.containers):
            self.increment_visibility()

            byte_det = self.save_bytes()
            byte_details.append(byte_det)

        return byte_details


if __name__ == "__main__":
    breakpoint()
