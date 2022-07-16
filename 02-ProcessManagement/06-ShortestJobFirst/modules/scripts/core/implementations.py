from .enumerations import SchedulingType, EventType, ProcessState
import pandas as pd

import os
import io
import random
import string

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

    def prepare_processes(self, processes):
        ## Set Process as index
        processes = processes.set_index("Process")

        ## Drop Priority column if existing
        if "Priority No." in processes.columns:
            processes = processes.drop(columns="Priority No.")

        ## Sort by chronological order for proper simulation
        processes = processes.sort_values(by="Arrival Time")

        ## Add new columns
        processes[["Start", "Finish"]] = -1

        return processes

    def validate_scheduling(self, input_type):
        ## Condition: Scheduling type must be supported
        supported = self.validations["scheduling"]
        msg = f"Scheduling type must be one of {supported}; got {input_type}."
        assert input_type in supported, msg


class Processor:
    """
    GOAL: Model CPU for clarity
    """

    in_use = False
    process = {  ## TODO: Better way?
        "Name": None,
        "Release": None,
    }

    # queue: list[str] = None  ## TODO: Transform into dataframe instead
    queue: pd.DataFrame
    queue_cols = ["Process", "Arrival Time", "Burst Time", "Priority No."]

    finished: list[str] = None

    def __init__(self):
        self.queue = pd.DataFrame(columns=self.queue_cols)
        # self.queue = []

        self.finished = []

    def __repr__(self):
        cls = self.__class__.__name__
        process = self.process
        queue = self.queue
        finished = self.finished

        return f"{cls}({process=}, {queue=}, {finished=})"

    def enqueue(self, proc, queue_time, remaining_time, priority_level):
        q = self.queue
        q.loc[len(q)] = (proc, queue_time, remaining_time, priority_level)

        # self.queue.append(proc)

    def dequeue(self):
        row0 = self.queue.iloc[0]
        self.queue = self.queue.iloc[1:].reset_index(drop=True)
        return row0

        # return self.queue.pop(0)


class Simulation:
    """
    GOAL: Produce "characteristic" table
    GOAL: Create timepoints
    """

    ## Simulation time properties initialized in "actual" simulation
    clock: int
    end: int  # Running value; changed with each process dispatch
    # idle: int  ## TODO: Track this in CPU object instead? Or delete?
    is_idling: bool

    ## Data
    processor: Processor
    processes: pd.DataFrame
    processes_full: pd.DataFrame
    idle_label = "<idle>"

    ## For creating an [event | timepoint] list
    events: pd.DataFrame
    events_cols = ["Process", "Time", "Type"]
    init_label = "<init>"

    timepoints: list[list]
    states: pd.DataFrame
    states_idxs = ["State", "Turnaround", "Waiting"]

    event_state_map = {
        EventType.Release: ProcessState.Terminated,
        EventType.Arrival: ProcessState.Ready,
        EventType.Dispatch: ProcessState.Running,
        EventType.Idling: ProcessState.Idle,
    }
    event_value_map = {
        EventType.Dispatch: "Waiting",
        EventType.Release: "Turnaround",
    }

    ## Debugging props
    ## TODO: Delete!
    DEBUG = True
    DEBUG_HARD_END = 100

    def __init__(self, processes: pd.DataFrame, scheduling_type):
        ## Assign instance attributes
        self.processor = Processor()
        self.processes = processes
        self.scheduling = scheduling_type

        ## Start simulation
        ## TODO: Remove this? Start explicitly?
        self.simulate()

        ## Add post-mortem values
        self.finalize()

    def __cpu_proc_dispatch(self, proc_name, start, runtime):
        cpu = self.processor

        ## Move simulation end mark
        ## TODO: Use this variable in the expressions below?
        self.end = start + runtime

        ## TODO: Mark process as RUNNING
        cpu.process["Name"] = proc_name
        cpu.process["Release"] = start + runtime
        cpu.in_use = True
        self.is_idling = False
        self.processes.at[proc_name, ["Start", "Finish"]] = (start, start + runtime)

        self.__event_add(proc_name, self.clock, EventType.Dispatch)

    def __cpu_proc_release(self):
        cpu = self.processor

        self.__event_add(cpu.process["Name"], self.clock, EventType.Release)

        cpu.in_use = False
        cpu.finished.append(cpu.process["Name"])
        for prop in cpu.process:
            cpu.process[prop] = None

    def __apply_scheduling(self, processes):
        """
        Apply scheduling logic
            FCFS - No additional side effect
            SJF  - Sort by sizes
            Prio - ...
            ...
        """

        FCFS, SJF, Priority, SRTF, RR = SchedulingType

        ## Reset index as column
        index_name = processes.index.name or "index"
        processes = processes.reset_index()

        if self.scheduling == FCFS:
            ## First come (First to arrive)
            processes = processes.sort_values(by=["Arrival Time", index_name])

        elif self.scheduling == SJF:
            processes = processes.sort_values(by=["Burst Time", index_name])

        # elif self.scheduling == Priority:
        #     ...

        # elif self.scheduling == SRTF:
        #     ...

        # elif self.scheduling == RR:
        #     ...

        else:
            raise NotImplementedError(f"Type {self.scheduling} not yet implemented.")

        ## Set index again
        processes = processes.set_index(index_name)

        return processes

    def __event_add(self, process, time, event_type):
        self.events.loc[len(self.events)] = process, time, event_type

        if event_type == EventType.Idling:
            process = " "

        ## Update process states
        if event_type in self.event_state_map:
            state = self.event_state_map[event_type]
            self.states.loc["State", process] = state

        if event_type in self.event_value_map:
            arrival = self.processes.at[process, "Arrival Time"]
            value_key = self.event_value_map[event_type]
            self.states.loc[value_key, process] = self.clock - arrival

        ## Set blocked processes
        ready_mask = self.states.iloc[0] == ProcessState.Ready
        ready_procs = self.states.columns[ready_mask]

        blocked_mask = self.processes.loc[ready_procs, "Arrival Time"] < self.clock
        blocked_procs = ready_procs[blocked_mask]

        self.states.loc["State", blocked_procs] = ProcessState.Blocked

        ## Customize states table
        states_custom = self.states.copy()
        states_custom = states_custom.fillna("...")
        states_custom.index.name = ""
        states_custom = states_custom.reset_index()

        # print(
        #     self.states,
        #     sep="\n",
        #     end="\n\n",
        # )

        ## Add timepoint
        """ Time | Process name | Event type | Process states | Figure (Constructed separately!) """
        # timepoint = [time, process, event_type, self.states.copy(), ...]
        timepoint = [time, process, event_type, states_custom]
        self.timepoints.append(timepoint)

    def __sim_arrivals(self):
        procs = self.processes
        cpu = self.processor

        ## Get arriving processes
        ## TODO: Apply scheduling logic
        arrivals = procs.loc[procs["Arrival Time"] == self.clock].copy()
        arrivals = self.__apply_scheduling(arrivals)

        ## Short-circuit: Do nothing if no jobs arrived
        if len(arrivals) == 0:
            return

        ## Create running iterable
        iterable = arrivals.itertuples()

        ## Get "first" process
        ## Also removes it from running iterable
        candidate = next(iterable)
        name_idx, burst, arrival, *_ = candidate
        self.__event_add(name_idx, self.clock, EventType.Arrival)

        ## TODO: Apply scheduling logic (Dispatch high priority first)
        if not cpu.in_use:
            ## If nothing is currently being processed, dispatch candidate process to CPU
            self.__cpu_proc_dispatch(
                proc_name=name_idx,
                start=self.clock,
                runtime=burst,
            )
        else:
            ## If something is currently being procesed, enqueue the candidate process
            ## TODO: Mark as READY
            ## TODO: Merge this with loop below?
            cpu.enqueue(name_idx, arrival, burst, 0)

        ## Remaining processes are sure to be enqueued for later consideration
        ## TODO: Mark process(es) as READY
        ## TODO: Create ARRIVAL event(s)
        for proc in iterable:
            name_idx, burst, arrival, *_ = proc
            self.__event_add(name_idx, arrival, EventType.Arrival)
            cpu.enqueue(name_idx, arrival, burst, 0)

    def __sim_endings(self):
        ## Short-circuit: Do nothing if...
        if (
            ## ...
            (not self.processor.in_use)
            or (self.clock != self.processor.process["Release"])
        ):
            return

        procs = self.processes
        cpu = self.processor

        ## Release process from CPU
        self.__cpu_proc_release()

        ## Consider waiting processes
        if len(cpu.queue) > 0:
            ## TODO: Apply scheduling logic on queue
            cpu.queue = self.__apply_scheduling(cpu.queue)

            ## Dispatch first process in queue
            # name_idx = cpu.dequeue()
            # burst, arrival, *_ = procs.loc[name_idx]

            name_idx, arrival, burst, priority = cpu.dequeue()

            self.__cpu_proc_dispatch(
                proc_name=name_idx,
                start=self.clock,
                runtime=burst,
            )

    def __sim_idle(self):
        ## Short-circuit: Do nothing if...
        if (
            ## ...
            self.processor.in_use
            or (self.clock == self.end)
        ):
            return

        # self.idle += 1
        if not self.is_idling:
            self.is_idling = True
            self.__event_add(self.idle_label, self.clock, EventType.Idling)

    def simulate(self):
        ## Initialize simulation properties
        self.clock = 0
        self.end = -1
        # self.idle = 0
        self.is_idling = False

        self.events = pd.DataFrame(columns=self.events_cols)
        self.timepoints = list()
        self.states = pd.DataFrame(index=self.states_idxs)

        # import time
        # start = time.time()
        # print(
        #     self.processes,
        #     sep="\n",
        # )

        ## "Before time 0..." | Initial state
        self.__event_add(self.init_label, -1, EventType.Init)

        while len(self.processor.finished) < len(self.processes):
            ## Consider arriving processes
            self.__sim_arrivals()

            ## Consider ending processes
            self.__sim_endings()

            ## Consider idle times
            self.__sim_idle()

            # ## [Continue | End] simulation
            # ## TODO: Change `while` condition? while len(cpu.finished) < len(procs)
            # if (
            #     ## ...
            #     self.clock == self.end
            #     or (self.DEBUG and self.clock == self.DEBUG_HARD_END)
            # ):
            #     print("hi")
            #     break
            self.clock += 1

        # print(
        #     time.time() - start,
        #     self.processes,
        #     self.clock,
        #     sep="\n",
        # )

    def finalize(self):
        ## Sort tables in chronological order
        self.processes = self.processes.sort_values(by="Start")

        ## Compute idle periods
        ## `last_finish` : Time since latest process finish with respect to current process
        ## Gets difference between process start and last process finish, i.e. "idle periods"
        last_finish = self.processes["Finish"].shift(periods=1, fill_value=0)
        self.processes["Idle"] = self.processes["Start"] - last_finish

        ## Create idle periods
        self.processes_full = self.__add_idle_periods()

        ## Compute Turnaround and Waiting times
        self.processes["Turnaround"] = (
            self.processes["Finish"] - self.processes["Arrival Time"]
        )
        self.processes["Waiting"] = (
            self.processes["Start"] - self.processes["Arrival Time"]
        )

        self.events["_"] = self.events["Type"].apply(lambda item: item.value)
        self.events = self.events.sort_values(by=["Time", "_"])
        self.events = self.events.drop(columns="_")

        # print(
        #     self.processes,
        #     (1 - (self.idle / self.end)) * 100,
        #     procs["Turnaround"].mean(),
        #     procs["Waiting"].mean(),
        #     sep="\n",
        # )

    def __add_idle_periods(self):
        procs = self.processes.copy().reset_index()
        idle_label = self.idle_label

        with_preceding_idles = procs[procs["Idle"] > 0]
        for proc in with_preceding_idles.itertuples():
            _, _, _, _, proc_start, _, idle_duration = proc

            idle_start = proc_start - idle_duration

            idle = [idle_label, idle_duration, idle_start, idle_start, proc_start, 0]
            procs.loc[len(procs)] = idle

        procs = procs.sort_values(by="Start")
        return procs


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

    def __init__(self, processes):
        ## Initialize attributes
        self.struct = processes.reset_index().copy()

        ## Prepare representative structure
        self.__prepare_repr_struct()

        ## Draw bar plot
        self.__draw_plot()

    def __prepare_repr_struct(self):
        ## Initialize colors column
        self.struct["Color"] = random.sample(self.color_selection, k=len(self.struct))

        ## Reduce columns
        _ = ["Process", "Burst Time", "Color", "Arrival Time", "Idle", "Start"]
        self.struct = self.struct[_]

        ## Add idle periods
        with_preceding_idles = self.struct[self.struct["Idle"] > 0]
        for proc in with_preceding_idles.itertuples():
            _, _, _, _, proc_arriv, idle_duration, _ = proc
            idle_start = proc_arriv - idle_duration

            _ = [self.idle_label, idle_duration, self.neutral_color, idle_start, 0, 0]
            self.struct.loc[len(self.struct)] = _

        self.struct = self.struct.sort_values(by="Start")

        ## Perform label truncation accordingly
        rel_pcts = self.struct["Burst Time"] / self.struct["Burst Time"].sum()
        self.struct["pcts"] = rel_pcts

        candidate_mask = self.struct["pcts"] < self.trunc_thresh
        self.struct.loc[candidate_mask, "Process"] = ""  # "Remove" by setting as blank

        self.struct = self.struct.drop(columns="pcts")

        ## Reset index counts
        self.struct = self.struct.reset_index(drop=True)

        ## Keep only relevant columns
        self.struct = self.struct[["Process", "Burst Time", "Color"]]

        ## Add first blank bar, for helping bar labels
        self.struct.index += 1
        self.struct.loc[0] = "", 0, self.neutral_color
        self.struct = self.struct.sort_index()

        ## "Pivot", for stacked plotting
        self.struct = self.struct.T

    def __draw_plot(self):
        ## Plot Gantt chart
        gantt = self.struct.loc[["Burst Time"]].plot(
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

    def toggle_visibility(self, to_on=False):
        self.visible_bars = 0

        for idx, container in enumerate(self.gantt.containers):
            ## Hide bars
            container.patches[0].set_alpha(0)

            ## Hide text labels
            self.annotations[idx * 2].set_alpha(0)
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
        self.toggle_visibility(to_on=False)

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
        self.toggle_visibility(to_on=False)

        byte_details = []
        for idx, container in enumerate(self.gantt.containers):
            self.increment_visibility()

            byte_det = self.save_bytes()
            byte_details.append(byte_det)

        return byte_details
