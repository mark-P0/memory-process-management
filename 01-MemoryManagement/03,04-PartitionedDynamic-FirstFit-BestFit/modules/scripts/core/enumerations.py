from enum import Enum


## For properly displaying enumeration names
Enum.__str__ = lambda self, *args: self.name

## Values
class PartitionStatus(Enum):
    Allocated = 0
    Available = 1


class EventType(Enum):
    Deallocation = 0
    Compaction = 1
    Arrival = 2
    Allocation = 3


class DynamicType(Enum):
    FirstFit = 0
    BestFit = 1


## Table columns
class JTColumn(Enum):
    JobNo = "Job No."
    Size = "Size"
    Arrival = "Arrival Time"
    Runtime = "Run Time (min.)"


class PATColumn(Enum):
    PartitionNo = "Partition No."
    Size = "Size"
    Location = "Location"


class STColumn(Enum):
    JobNo = "Job No."
    Arrival = "Arrival"
    Start = "Start"
    Finish = "Finish"
    CPUWait = "CPU Wait (min.)"
    IsWaiting = "Is Waiting"  # For determining waiting states


class EventColumn(Enum):
    JobNo = "Job No."
    Time = "Time"
    Type = "Type"
    Partition = "Partition"
    TypeValues = "_Type_vals"


if __name__ == "__main__":
    import random

    # test = [
    #     ## ...
    #     PATColumn.PartitionNo,
    #     [col.value for col in PATColumn],
    #     list(STColumn),
    # ]
    # print(test)

    seq = random.choices(list(EventType), k=10)
    seq_vals = [item.value for item in seq]

    print(sorted(seq, key=lambda item: item.value))
    print(sorted(seq_vals))
