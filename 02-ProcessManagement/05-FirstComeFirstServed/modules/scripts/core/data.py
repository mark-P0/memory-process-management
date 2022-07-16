# fmt: off

LECTURE_EXAMPLES = {
    "Lecture_FCFS_SJF_1": {
        "Process":      ["P1", "P2", "P3"],
        "Burst Time":   [   7,   10,    5],
        "Arrival Time": [   5,    0,    8],
    },
    "Lecture_FCFS_SJF_2": {
        "Process":      ["P1", "P2", "P3", "P4"],
        "Burst Time":   [  10,    1,    2,    5],
        "Arrival Time": [  10,    5,    8,    6],

    },
    "Lecture_Prio": {
        "Process":      ["P1", "P2", "P3", "P4"],
        "Burst Time":   [  12,   10,    5,    7],
        "Arrival Time": [   4,    5,   10,    7],
        "Priority No.": [   2,    1,    4,    3],  # Extra column
    },
    "Lecture_SRTF_RR": {
        "Process":      ["P1", "P2", "P3", "P4"],
        "Burst Time":   [  12,   10,    5,    7],
        "Arrival Time": [   4,    5,   10,    7],
    },
    "Homework": {
        "Process":      ["P1", "P2", "P3", "P4", "P5", "P6"],
        "Burst Time":   [  25,   15,   27,   10,   13,   20],
        "Arrival Time": [   3,   10,    5,    6,    8,    4],
        "Priority No.": [   2,    4,    3,    6,    1,    5],
    },
    "Finals": {
        "Process":      ["P1", "P2", "P3", "P4", "P5", "P6"],
        "Burst Time":   [  25,   10,   20,   12,    5,   15],
        "Arrival Time": [   3,   23,   17,   35,   40,    4],
        "Priority No.": [   2,    6,    1,    5,    3,    4],
    },


}

# fmt: on
