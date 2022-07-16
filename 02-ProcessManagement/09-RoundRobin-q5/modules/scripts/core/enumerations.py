from enum import Enum


## For properly displaying enumeration names
Enum.__str__ = lambda self, *args: self.name


class SchedulingType(Enum):
    FCFS = "First Come, First Served"
    SJF = "Shortest Job First"
    Priority = "Priority"
    SRTF = "Shortest Remaining Time First"
    RR = "Round Robin"


class EventType(Enum):
    Init = -1
    Idling = 0
    Release = 1
    Arrival = 2
    Dispatch = 3

    Suspend = 4
    Resume = 5


class ProcessState(Enum):
    Terminated = 0
    Running = 1
    Blocked = 2
    Ready = 3
    Idle = 4


class ProcessColumns(Enum):
    Process = "Process"
    Burst = "Burst Time"
    Arrival = "Arrival Time"

    Priority = "Priority No."

    Start = "Start Time"
    Finish = "Finish Time"
    Turnaround = "Turnaround"
    Waiting = "Waiting"

    Color = "Color"


if __name__ == "__main__":
    # res = [
    #     list(ProcessColumns),
    # ]
    # print(res)

    breakpoint()
